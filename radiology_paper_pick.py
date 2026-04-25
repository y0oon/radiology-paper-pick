"""
🏥 Radiology Paper Pick - 放射線科論文自動ピックアップシステム
--------------------------------------------------
PubMedから最新の放射線科論文を取得し、Geminiで要約、WordPressへ自動投稿します。
"""

import os
import json
import yaml
import requests
import datetime
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from dotenv import load_dotenv
import google.generativeai as genai

# ==========================================
# 1. 設定の読み込みと初期化 ⚙️
# ==========================================

# .envから強制的に最新の設定を読み込み
load_dotenv(override=True)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

NCBI_API_KEY = os.environ.get("NCBI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
WP_USER = os.environ.get("WP_USER")
WP_APP_PASS = os.environ.get("WP_APP_PASS")

def initialize_gemini():
    """Gemini AIの初期化とモデルの自動選択"""
    if not GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY が設定されていません。")
        return None
        
    genai.configure(api_key=GEMINI_API_KEY)
    
    try:
        # 最新のモデル（2.0-flash）を最優先で使用
        model_name = config.get("gemini", {}).get("model", "gemini-2.0-flash")
        full_model_name = f"models/{model_name}" if not model_name.startswith("models/") else model_name
        return genai.GenerativeModel(full_model_name)
    except Exception as e:
        print(f"Gemini初期化エラー: {e}")
        return genai.GenerativeModel("models/gemini-2.0-flash")

model = initialize_gemini()

# ==========================================
# 2. 論文データの取得 📚
# ==========================================

def fetch_pubmed_ids(query: str, max_results: int = 15) -> List[str]:
    """検索クエリに合致するPMIDのリストを取得"""
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
    
    try:
        response = requests.get(base_url, params=params, timeout=20)
        response.raise_for_status()
        return response.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"PubMed ID取得失敗: {e}")
        return []

def fetch_pubmed_details(id_list: List[str]) -> List[Dict]:
    """PMIDリストから論文の詳細（タイトル、抄録等）を取得"""
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
        
    try:
        response = requests.get(base_url, params=params, timeout=30)
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
    except Exception as e:
        print(f"PubMed詳細取得失敗: {e}")
        return []

def score_paper(paper: Dict, priority_journals: List[str], keywords: List[str]) -> float:
    """論文の重要度をスコアリング"""
    score = 0.0
    journal = str(paper.get("journal") or "").lower()
    title = str(paper.get("title") or "").lower()
    
    # 優先雑誌（Radiologyなど）への加点
    if any(pj.lower() in journal for pj in priority_journals):
        score += 10.0
    
    # テーマキーワードへの加点
    for kw in keywords:
        if kw.lower() in title:
            score += 2.0
            
    return score

# ==========================================
# 3. AIによる要約生成 🤖
# ==========================================

def generate_ai_summary(paper: Dict, theme_name: str, assistant_config: Dict) -> Optional[Dict]:
    """Geminiを使用して論文を日本語要約"""
    if not model:
        return None

    prompt = f"""You are {assistant_config['name']}, {assistant_config['role']}.
Your task: Summarize the provided medical paper abstract in JAPANESE.

[MANDATORY REQUIREMENTS]
1. Respond ONLY in JAPANESE.
2. Create a catchy JAPANESE title.
3. Provide a summary in JAPANESE (maximum 3 bullet points).
4. Provide expert insight in JAPANESE (approx. 100 characters).
5. Provide SEO description and keywords in JAPANESE.
6. Return the response strictly in JSON format.

[Output Format (JSON)]
{{
  "jp_title": "日本語のタイトル",
  "summary": ["ポイント1", "ポイント2", "ポイント3"],
  "eye_content": "専門的コメント",
  "seo_description": "SEO説明文",
  "keywords": ["キーワード1", "キーワード2"]
}}

[Paper Details]
Title: {paper['title']}
Journal: {paper['journal']}
Abstract: {paper['abstract']}
"""
    # 429エラー対策のリトライループ
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt, 
                generation_config={"response_mime_type": "application/json"}
            )
            result = json.loads(response.text, strict=False)
            
            if isinstance(result.get("summary"), str):
                result["summary"] = [result["summary"]]
                
            if "jp_title" in result and "summary" in result:
                return result
            return None

        except Exception as e:
            if "429" in str(e):
                wait_time = (attempt + 1) * 35
                print(f"⚠️ 制限に達しました。{wait_time}秒待機して再試行します... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"AI要約失敗 (PMID {paper['pmid']}): {e}")
                break
    return None

# ==========================================
# 4. HTML・コンテンツ作成 🎨
# ==========================================

def construct_html(summaries: List[Dict], theme_name: str, assistant_config: Dict) -> str:
    """WordPress投稿用のHTMLを構築（Gutenbergブロック対応）"""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    html = "<!-- wp:html -->\n"
    html += "<div class=\"paper-pick-container\" style=\"font-family: 'Helvetica Neue', Arial, sans-serif; color: #333; line-height: 1.8; max-width: 850px; margin: 0 auto;\">\n"
    
    header_style = "background: linear-gradient(135deg, #1a2a6c 0%, #b21f1f 50%, #fdbb2d 100%); color: white; padding: 40px; border-radius: 15px; margin-bottom: 40px; box-shadow: 0 10px 20px rgba(0,0,0,0.1);"
    html += f"""<div class="hero-card" style="{header_style}">
        <div style="font-size: 2.2em; font-weight: bold; margin-bottom: 10px;">Recommended Papers</div>
        <div style="font-size: 1.5em; border-bottom: 2px solid rgba(255,255,255,0.4); padding-bottom: 10px; margin-bottom: 15px;">本日のテーマ: {theme_name}</div>
        <div style="font-size: 1.1em; opacity: 0.95;">最新の重要論文{len(summaries)}件をピックアップ（{today} 更新）</div>
    </div>"""

    for item in summaries:
        paper, ai = item["paper"], item["ai_data"]
        html += f"""<div class="paper-item" style="background: #fff; border: 1px solid #e0e0e0; border-left: 8px solid #1a2a6c; padding: 30px; margin-bottom: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
            <div style="font-size: 1.6em; font-weight: bold; color: #1a2a6c; margin-bottom: 15px; line-height: 1.4;">{ai['jp_title']}</div>
            <div style="font-size: 0.9em; color: #666; margin-bottom: 20px; background: #f8f9fa; padding: 10px; border-radius: 5px;">
                📖 <b>{paper['journal']}</b> | 🔗 <a href="{paper['url']}" target="_blank" style="color: #3498db; text-decoration: none;">PubMed原文を表示</a>
            </div>
            <ul style="margin-bottom: 25px; padding-left: 20px; list-style-type: square;">"""
        for s in ai["summary"]:
            html += f"<li style=\"margin-bottom: 10px;\">{s}</li>"
        html += f"""</ul>
            <div style="background: #fff9db; padding: 20px; border-radius: 8px; border: 1px solid #ffe066;">
                <div style="font-weight: bold; color: #f08c00; margin-bottom: 8px; font-size: 1.1em;">💡 {assistant_config['eye_label']}</div>
                <div style="font-style: italic; color: #444;">{ai['eye_content']}</div>
            </div>
        </div>"""
    
    html += """<div style="text-align: center; margin-top: 50px;">
        <a href="/" class="rdt-footer" style="display: inline-block; padding: 15px 30px; background: #333; color: white; text-decoration: none; border-radius: 30px; font-weight: bold; transition: 0.3s;">Back to Home</a>
    </div>"""
    
    html += "\n</div>\n<!-- /wp:html -->"
    return html

# ==========================================
# 5. WordPress投稿 🌐
# ==========================================

def post_to_wordpress(title: str, content: str, excerpt: str = ""):
    """WordPress REST APIを使用して投稿を作成"""
    if not (WP_USER and WP_APP_PASS):
        print("❌ 認証情報が設定されていません。")
        return
        
    endpoint = config["wordpress"]["endpoint"]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "application/json"
    }
    auth = (WP_USER, WP_APP_PASS)

    print(f"🔑 WordPress認証テスト中... ({WP_USER})")
    try:
        base_url = endpoint.split("/wp-json/")[0]
        test_res = requests.get(f"{base_url}/wp-json/wp/v2/users/me", auth=auth, headers=headers, timeout=15)
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
        "status": config["wordpress"].get("status", "draft"),
        "categories": [config["wordpress"]["category_id"]],
        "excerpt": excerpt
    }
    
    print("📝 投稿処理を実行中...")
    try:
        response = requests.post(endpoint, json=data, auth=auth, headers=headers, timeout=30)
        if response.status_code in [200, 201]:
            print(f"✅ 投稿成功！ ID: {response.json().get('id')}")
            print(f"🔗 URL: {response.json().get('link')}")
        else:
            print(f"❌ 投稿失敗: {response.status_code} - {response.text[:200]}")
    except Exception as e:
        print(f"❌ 通信エラー: {e}")

# ==========================================
# 6. メイン実行 🚀
# ==========================================

def main():
    weekday = datetime.datetime.now().weekday()
    theme = config["daily_themes"][weekday]
    print(f"=== 本日のテーマ: {theme['name']} ===")
    
    ids = fetch_pubmed_ids(theme["query"])
    if not ids:
        print("論文が見つかりませんでした。")
        return
    
    papers = fetch_pubmed_details(ids)
    for p in papers:
        p["score"] = score_paper(p, config["priority_journals"], theme["keywords"])
    
    papers.sort(key=lambda x: x["score"], reverse=True)
    valid_summaries = []
    
    for p in papers[:5]:
        print(f"要約生成中: {p['title'][:60]}...")
        ai_data = generate_ai_summary(p, theme["name"], config["assistant"])
        if ai_data:
            valid_summaries.append({"paper": p, "ai_data": ai_data})
            # 無料枠のレート制限回避のため、長めに待機
            time.sleep(35) 
        
    if not valid_summaries:
        print("有効な要約を生成できませんでした。")
        return

    html = construct_html(valid_summaries, theme["name"], config["assistant"])
    main_ai = valid_summaries[0]["ai_data"]
    
    today_str = datetime.datetime.now().strftime("%m/%d")
    title = f"【{today_str}】{theme['name']} 放射線科最新論文ピックアップ (Top {len(valid_summaries)})"
    
    post_to_wordpress(title, html, main_ai.get("seo_description", ""))

if __name__ == "__main__":
    main()

# ==========================================
# 🎁 以下は参考用コードです（通常は実行されません）
# ==========================================
def send_email_reference(subject: str, html_content: str):
    """【参考】HTMLメールを送信する関数"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    email_cfg = config.get("email_settings", {})
    email_to = email_cfg.get("to")
    email_from = email_cfg.get("from")
    smtp_server = email_cfg.get("smtp_server")
    smtp_port = email_cfg.get("smtp_port")
    smtp_pass = os.environ.get("SMTP_PASS")

    if not (email_from and smtp_pass):
        print("❌ メールの認証情報（email_from, SMTP_PASS）が設定されていません。")
        return

    msg = MIMEMultipart()
    msg["From"] = email_from
    msg["To"] = email_to
    msg["Subject"] = subject
    msg.attach(MIMEText(html_content, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_from, smtp_pass)
            server.send_message(msg)
        print("✅ メール送信成功！")
    except Exception as e:
        print(f"❌ メール送信失敗: {e}")

