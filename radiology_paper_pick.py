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

# ==========================================
# 1. 設定の読み込みと初期化 ⚙️
# ==========================================

# 1-1. config.yaml（設定ファイル）を読み込みます
# このファイルには、検索ワードやAIのキャラクター設定などが書かれています
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 1-2. 環境変数から機密情報（APIキーなど）を取り出します
# セキュリティのため、コード内に直接パスワードを書かず、OSの「環境変数」から取得するのが一般的です
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")     # PubMed(NCBI)用の鍵
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Gemini AI用の鍵
WP_USER = os.environ.get("WP_USER")               # WordPressのユーザー名
WP_APP_PASS = os.environ.get("WP_APP_PASS")       # WordPressのアプリパスワード

# 1-3. Gemini AIの準備をします
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(config["gemini"]["model"])

# ==========================================
# 2. PubMed（論文データベース）からのデータ取得 📚
# ==========================================

def fetch_pubmed_ids(query: str, max_results: int = 20) -> List[str]:
    """
    検索ワード（query）に一致する論文のID（PMID）を最新2日分取得します。
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": f"{query} AND (\"last 2 days\"[Filter])", # 2日以内の論文に限定
        "retmode": "json",                               # 結果をJSONで受け取る
        "retmax": max_results                            # 最大取得件数
    }
    # APIキーがあれば追加（制限が緩和されます）
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    
    # 実際にインターネットでリクエストを送ります
    response = requests.get(base_url, params=params)
    response.raise_for_status() # エラーがあればここで停止させる
    data = response.json()
    
    # 見つかった論文のIDリストを返します
    return data.get("esearchresult", {}).get("idlist", [])

def fetch_pubmed_details(id_list: List[str]) -> List[Dict]:
    """
    論文IDのリストを受け取り、それぞれのタイトル、掲載雑誌、抄録（要約）を取得します。
    """
    if not id_list:
        return []
        
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml" # 詳細情報はXML形式でしか取れない項目が多いためXMLを使用
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
        
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    
    # XMLを解析して必要なデータだけを取り出します
    root = ET.fromstring(response.content)
    papers = []
    for article in root.findall(".//PubmedArticle"):
        title = article.find(".//ArticleTitle").text if article.find(".//ArticleTitle") is not None else "No Title"
        journal = article.find(".//Title").text if article.find(".//Title") is not None else "No Journal"
        abstract_node = article.find(".//AbstractText")
        abstract = abstract_node.text if abstract_node is not None else "No Abstract"
        pmid = article.find(".//PMID").text
        
        # 扱いやすいように辞書形式にまとめます
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
    # 有名な雑誌（Radiologyなど）に掲載されている場合はボーナス点
    if any(pj.lower() in paper["journal"].lower() for pj in priority_journals):
        score += 10.0
    
    # タイトルに特定のキーワードが含まれている場合も加点
    for kw in keywords:
        if kw.lower() in paper["title"].lower():
            score += 2.0
            
    return score

# ==========================================
# 4. Gemini AIによる要約生成 🤖
# ==========================================

def generate_ai_summary(paper: Dict, theme_name: str, assistant_config: Dict) -> Dict:
    """
    英語の論文情報をGeminiに渡し、日本語のタイトル、要約、専門的コメントを生成します。
    """
    # AIへの指示書（プロンプト）を組み立てます
    prompt = f"""You are the "{assistant_config['name']}" ({assistant_config['role']}).
提供された抄録に基づいて、医療従事者や学生向けの日本語要約を作成してください。
本日のテーマは「{theme_name}」です。

[要件]
1. キャッチーな日本語タイトルを作成（直訳ではなく意図を汲むこと）。
2. 3点以内の箇条書きで日本語要約を提供。
3. "{assistant_config['eye_label']}"として、専門的な視点やアドバイスを日本語で提供（100文字程度）。
4. 口調は「{assistant_config['tone']}」で。
5. 追加指示: {assistant_config['instruction']}

論文詳細:
Title: {paper['title']}
Journal: {paper['journal']}
Abstract: {paper['abstract']}

出力形式 (JSONのみ):
{{
  "jp_title": "日本語タイトル",
  "summary": ["ポイント1", "ポイント2", "ポイント3"],
  "eye_content": "専門的なコメント"
}}
"""
    try:
        # AIに問い合わせます。JSON形式で返却するよう指定しています。
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        # AIの生成に失敗した場合のバックアップ処理
        print(f"Gemini Error for {paper['pmid']}: {e}")
        return {
            "jp_title": paper["title"],
            "summary": ["要約の生成に失敗しました。"],
            "eye_content": "詳細は原文をご確認ください。"
        }

# ==========================================
# 5. HTML（ブログ記事の見た目）の作成 🎨
# ==========================================

def construct_html(summaries: List[Dict], theme_name: str, assistant_config: Dict) -> str:
    """
    生成されたデータをWordPressに貼り付けられるHTML形式に整えます。
    """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    header_style = "background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 30px;"
    
    # ブロックエディタ(Gutenberg)用のコメントも含めてHTMLを組み立てます
    html = f"""<!-- wp:html -->
<div class="paper-pick-article">
    <header style="{header_style}">
        <h2 style="color: white; margin-top: 0;">{assistant_config['name']}'s Select: {theme_name} 最新論文</h2>
        <p style="margin-bottom: 0;">{theme_name}に関する注目の5論文をピックアップしました（{today}更新）</p>
    </header>
    
    <div class="paper-list">
"""
    
    for item in summaries:
        paper = item["paper"]
        ai_data = item["ai_data"]
        
        html += f"""
        <article class="paper-item" style="border-left: 4px solid #2c3e50; padding-left: 20px; margin-bottom: 40px;">
            <h3 style="color: #2c3e50; font-size: 1.4em;">{ai_data['jp_title']}</h3>
            <div class="meta" style="font-size: 0.9em; color: #666; margin-bottom: 10px;">
                掲載誌: <strong>{paper['journal']}</strong> | <a href="{paper['url']}" target="_blank">PubMedで見る</a>
            </div>
            <ul class="summary-list" style="margin-bottom: 15px;">
"""
        for s in ai_data["summary"]:
            html += f"                <li>{s}</li>\n"
            
        html += f"""            </ul>
            <div class="expert-insight" style="background: #f8f9fa; padding: 15px; border-radius: 5px; font-style: italic; border: 1px solid #e9ecef;">
                <strong style="color: #e74c3c;">{assistant_config['eye_label']}:</strong> {ai_data['eye_content']}
            </div>
        </article>
"""
    
    html += f"""
    </div>
    <footer style="margin-top: 40px; border-top: 1px solid #eee; padding-top: 20px; text-align: center; color: #999; font-size: 0.8em;">
        <p>この記事はAIにより自動生成されています。</p>
    </footer>
</div>
<!-- /wp:html -->
"""
    return html

# ==========================================
# 6. WordPressへの投稿 🌐
# ==========================================

def post_to_wordpress(title: str, content: str):
    """
    WordPress REST APIを使って、完成した記事を公開（または下書き保存）します。
    """
    if not WP_USER or not WP_APP_PASS:
        print("WordPressの認証情報が設定されていません。投稿をスキップします。")
        return
        
    auth = (WP_USER, WP_APP_PASS)
    data = {
        "title": title,
        "content": content,
        "status": config["wordpress"]["status"],        # 'publish'なら即公開、'draft'なら下書き
        "categories": [config["wordpress"]["category_id"]]
    }
    
    response = requests.post(config["wordpress"]["endpoint"], json=data, auth=auth)
    if response.status_code == 201:
        print(f"投稿成功: {title}")
    else:
        print(f"投稿失敗: {response.status_code}")
        print(response.text)

# ==========================================
# メイン処理の実行フロー 🚀
# ==========================================

def main():
    # A. 準備：今日が何曜日かを確認し、テーマ（CT、MRIなど）を決定します
    weekday = datetime.datetime.now().weekday()
    theme = config["daily_themes"][weekday]
    assistant_config = config["assistant"]
    
    print(f"=== 本日のテーマ: {theme['name']} ===")
    
    # 1. PubMedで論文のIDを探します
    ids = fetch_pubmed_ids(theme["query"])
    if not ids:
        print("条件に合う論文が見つかりませんでした。")
        return
        
    # 2. 見つかった論文の詳細情報を取得します
    papers = fetch_pubmed_details(ids)
    
    # 3. 各論文をスコア付けし、上位5件に絞り込みます
    for p in papers:
        p["score"] = score_paper(p, config["priority_journals"], theme["keywords"])
    
    papers.sort(key=lambda x: x["score"], reverse=True) # スコアの高い順に並び替え
    top_papers = papers[:5]                            # 上位5つを取得
    
    # 4. 上位論文を1つずつAIで要約します
    summaries = []
    for p in top_papers:
        print(f"要約生成中: {p['title'][:50]}...")
        ai_data = generate_ai_summary(p, theme["name"], assistant_config)
        summaries.append({"paper": p, "ai_data": ai_data})
        time.sleep(1) # APIの負荷を減らすため1秒待ちます
        
    # 5. すべての要約を1つのHTMLにまとめます
    html_content = construct_html(summaries, theme["name"], assistant_config)
    
    # 6. WordPressに記事を投稿します
    today_str = datetime.datetime.now().strftime("%m/%d")
    post_title = f"【{today_str}】{theme['name']} 放射線科最新論文ピックアップ (Top 5)"
    
    post_to_wordpress(post_title, html_content)


# このプログラムが直接実行された場合にのみ、main()を呼び出します
if __name__ == "__main__":
    main()
