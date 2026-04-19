# Radiology Paper Pick

Automated system to fetch the latest radiology papers from PubMed, summarize them using Gemini AI, and post the results to WordPress.

## Features
- **Daily Themes:** Automatically switches between 7 radiology-related themes (CT, MRI, Nuclear Medicine, etc.) based on the day of the week.
- **AI Summarization:** Uses Gemini AI to generate professional Japanese summaries and insights for healthcare professionals.
- **Automated Workflow:** Runs daily via GitHub Actions.
- **WordPress Integration:** Posts formatted HTML directly to your WordPress site.

## Setup

### 1. Requirements
You will need the following API keys and credentials:
- **Gemini API Key:** Obtain from [Google AI Studio](https://aistudio.google.com/).
- **NCBI API Key:** Obtain from your [PubMed](https://pubmed.ncbi.nlm.nih.gov/) account settings (recommended for higher rate limits).
- **WordPress App Password:** Generate from your WordPress User Profile settings.

### 2. GitHub Secrets
Register the following secrets in your repository under `Settings > Secrets and variables > Actions`:
- `GEMINI_API_KEY`
- `NCBI_API_KEY`
- `WP_USER`
- `WP_APP_PASS`

### 3. Customization
Edit `config.yaml` to customize:
- Priority journals
- Search queries and keywords for each day
- WordPress endpoint and category ID
- AI Assistant persona and tone

## Credits / Special Thanks
The core logic of this system (PubMed scoring and summarization flow) is based on the following project by **yush02084**:

- [medical-paper-summarizer-public (GitHub)](https://github.com/yush02084/medical-paper-summarizer-public)

Special thanks to the original developer for sharing their wisdom.

## License
Distributed under the MIT License. See `LICENSE` (if any) for more information.
