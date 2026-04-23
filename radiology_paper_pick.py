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
def initialize_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    
    try:
        # 利用可能なモデルを一覧取得します
        all_models = list(genai.list_models())
        available_models = [m.name for m in all_models if "generateContent" in m.supported_generation_methods]
        
        # デバッグ用：現在使えるモデルをすべて表示（原因究明のため）
        # print(f"DEBUG: 利用可能なモデルリスト: {available_models}")
        
        # 1. 1.5-flash を最優先で探します（2.x系は除外）
        target_model = None
        for m_name in available_models:
            low_name = m_name.lower()
            if "1.5-flash" in low_name and "2.0" not in low_name and "2.5" not in low_name:
                target_model = m_name
                break
        
        # 2. 見つからない場合は、1.5-pro を探します
        if not target_model:
            for m_name in available_models:
                if "1.5-pro" in m_name.lower():
                    target_model = m_name
                    break

        # 3. それでも見つからない場合は、古い安定版（pro）を探します
        if not target_model:
            priority_list = ["models/gemini-1.0-pro", "models/gemini-pro"]
            for p in priority_list:
                if p in available_models:
                    target_model = p
                    break
        
        # 4. 最終手段：2.x系以外で最初に見つかったもの。なければリストの先頭。
        if not target_model:
            safe_models = [m for m in available_models if "2.0" not in m and "2.5" not in m]
            target_model = safe_models[0] if safe_models else available_models[0]
            
        print(f"DEBUG: 自動選択されたモデル: {target_model}")
        return genai.GenerativeModel(target_model)
        
    except Exception as e:
        print(f"Gemini初期化エラー（モデル取得失敗）: {e}")
        # 最終手段として直接指定
        return genai.GenerativeModel("models/gemini-pro")

model = initialize_gemini()

# ==========================================
# 2. PubMed（論文データベース）からのデータ取得 📚
# ==========================================

def fetch_pubmed_ids(query: str, max_results: int = 20) -> List[str]:
    """
    検索ワード（query）に一致する論文のID（PMID）を【最新順】で取得します。
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "sort": "date",                                  # 【重要】日付の新しい順に並び替える
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
    # 小規模モデルでも指示を外さないよう、英語で役割を与えつつ、日本語での出力を強く求めます
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
        # 1. AIに生成を依頼します
        # JSONモードを試みますが、失敗した場合は通常モードで取得します
        try:
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            res_text = response.text
        except Exception:
            response = model.generate_content(prompt)
            res_text = response.text

        # 2. 【超重要】JSONのクリーニング処理
        # Markdownのコードブロック（```json ... ```）を削除
        if "```" in res_text:
            res_text = res_text.split("```")
            for part in res_text:
                if "{" in part and "}" in part:
                    res_text = part.replace("json", "").strip()
                    break
            if isinstance(res_text, list): # 失敗した場合のフォールバック
                res_text = "".join(res_text)

        # 改行コードや制御文字（\n, \r, \tなど）が文字列内で悪さをしないように置換
        # ※ json.loadsは文字列内の生の改行を嫌うため、スペースに置き換えるなどの処理を行います
        import re
        # JSONの外側にある余計なテキストを排除
        res_text = re.search(r"\{.*\}", res_text, re.DOTALL)
        if res_text:
            res_text = res_text.group(0)
        else:
            print("⚠️ JSONの形式が見つかりませんでした。")
            return None

        # 制御文字（バックスラッシュなど）をエスケープ
        res_text = res_text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

        # 3. パースの実行
        result = json.loads(res_text, strict=False) # strict=False で多少の制御文字は許容する
        
        # 4. 【重要】型チェックと修正（1文字ずつ改行されるバグ対策）
        # summary が文字列なら、1つの要素を持つリストに変換します
        if "summary" in result:
            if isinstance(result["summary"], str):
                # 文字列を1文字ずつではなく、一つの塊として扱います
                result["summary"] = [result["summary"]]
            elif not isinstance(result["summary"], list):
                result["summary"] = [str(result["summary"])]
        
        # 必要な項目が含まれているか確認
        if "jp_title" in result and "summary" in result:
            return result
        else:
            print(f"⚠️ AIの応答に必要な項目が足りません。")
            return None

    except Exception as e:
        # AIの生成に失敗した場合はエラーを表示して None を返します
        print(f"AI Generation Error for {paper['pmid']}: {e}")
        # 最後の手段として、生のテキストを表示
        # print(f"DEBUG: Failed text: {res_text[:200]}...")
        return None

# ==========================================
# 5. HTML（ブログ記事の見た目）の作成 🎨
# ==========================================

def construct_html(summaries: List[Dict], theme_name: str, assistant_config: Dict) -> str:
    """
    生成されたデータをWordPressに貼り付けられるHTML形式に整えます。
    見出しタグ(h2, h3)を使わずdivタグとスタイルで構成することで、WordPressテーマのCSS干渉を防ぎます。
    """
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    # 全体のアウターコンテナ
    html = "<!-- wp:html -->\n"
    html += "<div class=\"paper-pick-container\" style=\"font-family: 'Helvetica Neue', Arial, sans-serif; color: #333; line-height: 1.6; max-width: 800px; margin: 0 auto;\">\n"

    # 1. ヒーローカード（冒頭の紹介セクション）
    header_style = "background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 40px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);"
    html += f"""
    <div class="hero-card" style="{header_style}">
        <div style="font-size: 1.8em; font-weight: bold; margin-bottom: 10px; line-height: 1.2;">
            {assistant_config['name']}'s Select
        </div>
        <div style="font-size: 1.4em; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 10px; margin-bottom: 15px;">
            本日のテーマ: {theme_name} 最新論文
        </div>
        <div style="font-size: 1em; opacity: 0.9;">
            {theme_name}に関する注目の{len(summaries)}論文をピックアップしました（{today} 更新）
        </div>
    </div>
    """

    # 2. 論文リストセクション
    for item in summaries:
        paper = item["paper"]
        ai_data = item["ai_data"]

        # 論文ごとのカード
        html += f"""
        <div class="paper-item" style="background: white; border: 1px solid #eee; border-left: 6px solid #2c3e50; border-radius: 8px; padding: 25px; margin-bottom: 35px; shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <!-- 日本語タイトル（見出し代わり） -->
            <div style="font-size: 1.4em; font-weight: bold; color: #2c3e50; margin-bottom: 15px; line-height: 1.4;">
                {ai_data['jp_title']}
            </div>

            <!-- メタ情報 -->
            <div class="meta" style="font-size: 0.85em; color: #7f8c8d; margin-bottom: 15px; border-bottom: 1px solid #f1f1f1; padding-bottom: 10px;">
                掲載誌: <span style="font-weight: bold; color: #34495e;">{paper['journal']}</span> | 
                <a href="{paper['url']}" target="_blank" style="color: #3498db; text-decoration: none; font-weight: bold;">[PubMedで原文を確認]</a>
            </div>

            <!-- AI要約リスト -->
            <ul class="summary-list" style="margin-bottom: 20px; padding-left: 20px; color: #444;">
        """
        for s in ai_data["summary"]:
            html += f"        <li style=\"margin-bottom: 8px;\">{s}</li>\n"

        # 専門家インサイト（Expert Insight）
        html += f"""
            </ul>
            <div class="expert-insight" style="background: #fdf2f2; padding: 18px; border-radius: 6px; border: 1px solid #fadbd8; position: relative;">
                <div style="font-weight: bold; color: #e74c3c; margin-bottom: 5px; display: flex; align-items: center;">
                    <span style="font-size: 1.2em; margin-right: 8px;">💡</span> {assistant_config['eye_label']}
                </div>
                <div style="font-style: italic; color: #555; font-size: 0.95em;">
                    {ai_data['eye_content']}
                </div>
            </div>
        </div>
        """

    # 3. フッター
    html += """
        <div style="margin-top: 50px; border-top: 2px dashed #eee; padding-top: 20px; text-align: center; color: #bdc3c7; font-size: 0.85em;">
            <p>※この記事は AI (Gemini 1.5 Flash) により自動生成されています。正確な情報は必ず原文をご確認ください。</p>
        </div>
    </div>
    <!-- /wp:html -->
    """
    return html
# ==========================================
# 6. WordPressへの投稿 🌐
# ==========================================

def post_to_wordpress(title: str, content: str, excerpt: str = "", keywords: List[str] = None, featured_image_id: int = None):
    """
    WordPress REST APIを使って、完成した記事を公開（または下書き保存）します。
    メタデータ（抜粋、カテゴリー、SEO用カスタムフィールド）およびアイキャッチ画像も含めて送信します。
    """
    if not WP_USER or not WP_APP_PASS:
        print("WordPressの認証情報が設定されていません。投稿をスキップします。")
        return
        
    auth = (WP_USER, WP_APP_PASS)
    
    # 投稿データの作成
    data = {
        "title": title,
        "content": content,
        "status": config["wordpress"]["status"],
        "categories": [config["wordpress"]["category_id"]],
        "excerpt": excerpt,
        "featured_media": featured_image_id if featured_image_id else None, # アイキャッチ画像ID
        # メタフィールド
        "meta": {
            "rank_math_description": excerpt,
            "_yoast_wpseo_metadesc": excerpt,
            "rank_math_focus_keyword": ", ".join(keywords) if keywords else ""
        }
    }
    
    # ※WordPressの設定によっては、API経由でのmeta更新にプラグインの許可が必要です
    
    response = requests.post(config["wordpress"]["endpoint"], json=data, auth=auth)
    if response.status_code == 201:
        print(f"投稿成功: {title}")
    else:
        # メタフィールドの送信でエラーが出る場合があるため、失敗したらメタなしで再試行
        print(f"メタ付き投稿に失敗（{response.status_code}）。メタなしで再試行します...")
        data.pop("meta", None)
        response = requests.post(config["wordpress"]["endpoint"], json=data, auth=auth)
        if response.status_code == 201:
            print(f"投稿成功（メタなし）: {title}")
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
    valid_summaries = []
    for p in top_papers:
        print(f"要約生成中: {p['title'][:50]}...")
        ai_data = generate_ai_summary(p, theme["name"], assistant_config)
        
        # エラー（None）でなければリストに追加します
        if ai_data is not None:
            valid_summaries.append({"paper": p, "ai_data": ai_data})
        else:
            print(f"⚠️ {p['pmid']} の要約生成に失敗したため、スキップします。")
            
        # 429エラーを避けるため、15秒間待機します（無料枠の制限対策）
        time.sleep(15) 
        
    # 5. すべての要約に失敗した場合は、投稿を中止します
    if not valid_summaries:
        print("全ての論文で要約の生成に失敗しました。投稿を中止します。")
        return

    # 6. すべての要約を1つのHTMLにまとめます
    html_content = construct_html(valid_summaries, theme["name"], assistant_config)
    
    # 7. メタデータ（抜粋・キーワード）の準備
    # 1番目の論文のデータを記事全体の代表データとして使用します
    main_ai_data = valid_summaries[0]["ai_data"]
    excerpt = main_ai_data.get("seo_description", "")
    keywords = main_ai_data.get("keywords", [])
    
    # 8. WordPressに記事を投稿します
    today_str = datetime.datetime.now().strftime("%m/%d")
    post_title = f"【{today_str}】{theme['name']} 放射線科最新論文ピックアップ (Top 5)"
    
    # テーマごとに設定された画像IDを取得（設定がなければNone）
    featured_image_id = theme.get("image_id")
    
    post_to_wordpress(post_title, html_content, excerpt, keywords, featured_image_id)


# このプログラムが直接実行された場合にのみ、main()を呼び出します
if __name__ == "__main__":
    main()
