"""
🏥 Radiology Paper Pick - 放射線科論文自動ピックアップシステム
--------------------------------------------------
このプログラムは、医学論文データベース「PubMed」から最新の放射線科関連の論文を取得し、
GoogleのAI（Gemini）を使って日本語で要約を作成、さらにWordPressへ自動投稿するシステムです。
"""

import os            # OSの機能（ファイルパス操作や環境変数の取得）を使うためのライブラリ
import json          # JSON形式のデータを扱うためのライブラリ
import yaml          # 設定ファイル（YAML）を読み込むためのライブラリ
import requests      # インターネット経由でデータを取得（HTTPリクエスト）するためのライブラリ
import datetime      # 日付や時刻を扱うためのライブラリ
import time          # 処理を一時停止（スリープ）させるためのライブラリ
import google.generativeai as genai  # Google Gemini AIを使うための公式ライブラリ
import xml.etree.ElementTree as ET   # XML形式のデータ（PubMedの応答）を解析するためのライブラリ
from typing import List, Dict        # コードの可読性を高めるための型ヒント（リストや辞書の指定）
from dotenv import load_dotenv       # .envファイルから環境変数を読み込むためのライブラリ

# ==========================================
# 1. 設定の読み込みと初期化 ⚙️
# ==========================================

# 1-0. .envファイルからAPIキーなどを読み込みます
load_dotenv()

# 1-1. config.yaml（設定ファイル）を読み込みます
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 1-2. 環境変数から機密情報（APIキーなど）を取り出します
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
WP_USER = os.environ.get("WP_USER")
WP_APP_PASS = os.environ.get("WP_APP_PASS")

# 1-3. Gemini AIの準備をします
def initialize_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    
    try:
        all_models = list(genai.list_models())
        available_models = [m.name for m in all_models if "generateContent" in m.supported_generation_methods]
        
        target_model = None
        for m_name in available_models:
            low_name = m_name.lower()
            if "1.5-flash" in low_name and "2.0" not in low_name and "2.5" not in low_name:
                target_model = m_name
                break
        
        if not target_model:
            for m_name in available_models:
                if "1.5-pro" in m_name.lower():
                    target_model = m_name
                    break

        if not target_model:
            priority_list = ["models/gemini-1.0-pro", "models/gemini-pro"]
            for p in priority_list:
                if p in available_models:
                    target_model = p
                    break
        
        if not target_model:
            safe_models = [m for m in available_models if "2.0" not in m and "2.5" not in m]
            target_model = safe_models[0] if safe_models else available_models[0]
            
        print(f"DEBUG: 自動選択されたモデル: {target_model}")
        return genai.GenerativeModel(target_model)
        
    except Exception as e:
        print(f"Gemini初期化エラー: {e}")
        return genai.GenerativeModel("models/gemini-pro")

model = initialize_gemini()

# ==========================================
# 2. PubMed（論文データベース）からのデータ取得 📚
# ==========================================

def fetch_pubmed_ids(query: str, max_results: int = 20) -> List[str]:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "sort": "date",
        "retmode": "json",
        "retmax": max_results
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])

def fetch_pubmed_details(id_list: List[str]) -> List[Dict]:
    if not id_list:
        return []
        
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml"
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
        
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    
    root = ET.fromstring(response.content)
    papers = []
    for article in root.findall(".//PubmedArticle"):
        title = article.find(".//ArticleTitle").text if article.find(".//ArticleTitle") is not None else "No Title"
        journal = article.find(".//Title").text if article.find(".//Title") is not None else "No Journal"
        abstract_node = article.find(".//AbstractText")
        abstract = abstract_node.text if abstract_node is not None else "No Abstract"
        pmid = article.find(".//PMID").text
        
        papers.append({
            "pmid": pmid,
            "title": title,
            "journal": journal,
            "abstract": abstract,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        })
    return papers

# ==========================================
# 3. 論文のスコアリング（重要度の判定） ⚖️
# ==========================================

def score_paper(paper: Dict, priority_journals: List[str], keywords: List[str]) -> float:
    """
    特定の雑誌やキーワードが含まれている場合にスコアを加算し、紹介する優先順位を決めます。
    """
    score = 0.0
    # 安全に文字列として取得
    journal = str(paper.get("journal") or "").lower()
    title = str(paper.get("title") or "").lower()
    
    # 1. 優先雑誌
    if any(pj.lower() in journal for pj in priority_journals):
        score += 10.0
    
    # 2. キーワード
    for kw in keywords:
        if kw.lower() in title:
            score += 2.0
            
    return score

# ==========================================
# 4. AIによる要約生成 🤖
# ==========================================

def generate_ai_summary(paper: Dict, theme_name: str, assistant_config: Dict) -> Dict:
    prompt = f"""You are a medical expert specialized in radiology.
Your task: Summarize the provided medical paper abstract in JAPANESE for doctors and radiographers.

[MANDATORY REQUIREMENTS]
1. Respond ONLY in JAPANESE. (必ず日本語で回答してください)
2. Create a catchy JAPANESE title.
3. Provide a summary in JAPANESE (maximum 3 bullet points).
4. Provide expert insight in JAPANESE (approx. 100 characters).
5. Provide SEO description and keywords in JAPANESE.
6. Return the response strictly in JSON format.

[Output Format (JSON)]
{{
  "jp_title": "日本語のタイトル",
  "summary": ["日本語のポイント1", "日本語のポイント2", "日本語のポイント3"],
  "eye_content": "日本語の専門的コメント",
  "seo_description": "日本語のSEO説明文",
  "keywords": ["キーワード1", "キーワード2"]
}}

[Paper Details]
Title: {paper['title']}
Journal: {paper['journal']}
Abstract: {paper['abstract']}
"""
    try:
        try:
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            res_text = response.text
        except Exception:
            response = model.generate_content(prompt)
            res_text = response.text

        if "```" in res_text:
            res_text = res_text.split("```")
            for part in res_text:
                if "{" in part and "}" in part:
                    res_text = part.replace("json", "").strip()
                    break
        
        import re
        match = re.search(r"\{.*\}", res_text, re.DOTALL)
        if match:
            res_text = match.group(0)
        
        res_text = res_text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        result = json.loads(res_text, strict=False)
        
        if "summary" in result:
            if isinstance(result["summary"], str):
                result["summary"] = [result["summary"]]
        
        if "jp_title" in result and "summary" in result:
            return result
        return None

    except Exception as e:
        print(f"AI Generation Error for {paper['pmid']}: {e}")
        return None

# ==========================================
# 5. HTML作成 🎨
# ==========================================

def construct_html(summaries: List[Dict], theme_name: str, assistant_config: Dict) -> str:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    html = "<!-- wp:html -->\n<div class=\"paper-pick-container\" style=\"font-family: Arial, sans-serif; color: #333; line-height: 1.6; max-width: 800px; margin: 0 auto;\">\n"
    
    header_style = "background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 40px;"
    html += f"""<div class="hero-card" style="{header_style}">
        <div style="font-size: 1.8em; font-weight: bold; margin-bottom: 10px;">{assistant_config['name']}'s Select</div>
        <div style="font-size: 1.4em; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 10px; margin-bottom: 15px;">本日のテーマ: {theme_name} 最新論文</div>
        <div style="font-size: 1em; opacity: 0.9;">{theme_name}に関する注目の{len(summaries)}論文をピックアップしました（{today} 更新）</div>
    </div>"""

    for item in summaries:
        paper, ai = item["paper"], item["ai_data"]
        html += f"""<div class="paper-item" style="background: white; border-left: 6px solid #2c3e50; padding: 25px; margin-bottom: 35px; border: 1px solid #eee; border-radius: 8px;">
            <div style="font-size: 1.4em; font-weight: bold; color: #2c3e50; margin-bottom: 15px;">{ai['jp_title']}</div>
            <div style="font-size: 0.85em; color: #7f8c8d; margin-bottom: 15px;">掲載誌: <b>{paper['journal']}</b> | <a href="{paper['url']}" target="_blank" style="color: #3498db;">[PubMedで確認]</a></div>
            <ul style="margin-bottom: 20px; padding-left: 20px;">"""
        for s in ai["summary"]:
            html += f"<li style=\"margin-bottom: 8px;\">{s}</li>"
        html += f"""</ul>
            <div style="background: #fdf2f2; padding: 18px; border-radius: 6px; border: 1px solid #fadbd8;">
                <div style="font-weight: bold; color: #e74c3c; margin-bottom: 5px;">💡 {assistant_config['eye_label']}</div>
                <div style="font-style: italic; color: #555;">{ai['eye_content']}</div>
            </div>
        </div>"""
    
    html += "</div><!-- /wp:html -->"
    return html

# ==========================================
# 6. WordPress投稿 🌐
# ==========================================

def post_to_wordpress(title: str, content: str, excerpt: str = "", keywords: List[str] = None, featured_image_id: int = None):
    if not (WP_USER and WP_APP_PASS):
        print("WordPressの認証情報が設定されていません。")
        return
        
    auth = (WP_USER, WP_APP_PASS)
    endpoint = config["wordpress"]["endpoint"]
    base_url = endpoint.split("/wp-json/")[0]
    headers = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}

    print("🔑 WordPress認証テスト中...")
    try:
        me_url = f"{base_url}/wp-json/wp/v2/users/me"
        test_res = requests.get(me_url, auth=auth, headers=headers, timeout=15)
        if test_res.status_code != 200:
            print(f"❌ 認証失敗 (Status: {test_res.status_code})")
            return
        print(f"✅ 認証成功: {test_res.json().get('name')}")
    except Exception as e:
        print(f"❌ 接続エラー: {e}")
        return

    data = {
        "title": title,
        "content": content,
        "status": "draft",
        "categories": [config["wordpress"]["category_id"]]
    }
    
    print("📝 投稿テスト中...")
    try:
        response = requests.post(endpoint, data=json.dumps(data), auth=auth, headers=headers, timeout=30)
        if response.status_code == 201:
            print("✅ 投稿成功！")
            post_id = response.json().get("id")
            update_data = {"status": config["wordpress"].get("status", "publish")}
            if excerpt: update_data["excerpt"] = excerpt
            if featured_image_id and int(featured_image_id) > 0: update_data["featured_media"] = int(featured_image_id)
            
            update_url = f"{endpoint}/{post_id}"
            requests.post(update_url, data=json.dumps(update_data), auth=auth, headers=headers, timeout=30)
            print(f"🎉 最終公開完了")
        else:
            print(f"❌ 投稿失敗: {response.status_code} - {response.text[:200]}")
    except Exception as e:
        print(f"❌ 通信エラー: {e}")

# ==========================================
# 7. メイン実行 🚀
# ==========================================

def main():
    weekday = datetime.datetime.now().weekday()
    theme = config["daily_themes"][weekday]
    print(f"=== 本日のテーマ: {theme['name']} ===")
    
    ids = fetch_pubmed_ids(theme["query"])
    if not ids: return
    
    papers = fetch_pubmed_details(ids)
    for p in papers:
        p["score"] = score_paper(p, config["priority_journals"], theme["keywords"])
    
    papers.sort(key=lambda x: x["score"], reverse=True)
    valid_summaries = []
    for p in papers[:5]:
        print(f"要約生成中: {p['title'][:50]}...")
        ai_data = generate_ai_summary(p, theme["name"], config["assistant"])
        if ai_data:
            valid_summaries.append({"paper": p, "ai_data": ai_data})
        time.sleep(15) 
        
    if not valid_summaries: return

    html = construct_html(valid_summaries, theme["name"], config["assistant"])
    main_ai = valid_summaries[0]["ai_data"]
    
    today_str = datetime.datetime.now().strftime("%m/%d")
    title = f"【{today_str}】{theme['name']} 放射線科最新論文ピックアップ (Top 5)"
    
    post_to_wordpress(title, html, main_ai.get("seo_description", ""), main_ai.get("keywords", []), theme.get("image_id"))

if __name__ == "__main__":
    main()
