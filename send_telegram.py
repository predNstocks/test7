"""
Send markdown report to Telegram.
Sends the text report and optionally the PDF file.
Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
"""
import os, sys, requests


def send_telegram_text(text):
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    if not bot_token or not chat_id:
        print('Telegram credentials not set. Skipping.')
        return

    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'Markdown',
    }

    resp = requests.post(url, json=payload, timeout=30)
    if resp.status_code == 200:
        print('Text message sent to Telegram.')
    else:
        print(f'Failed to send text: {resp.status_code} {resp.text}')


def send_telegram_file(file_path, caption=''):
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    if not bot_token or not chat_id:
        return

    url = f'https://api.telegram.org/bot{bot_token}/sendDocument'
    with open(file_path, 'rb') as f:
        files = {'document': f}
        data = {'chat_id': chat_id}
        if caption:
            data['caption'] = caption
        resp = requests.post(url, data=data, files=files, timeout=60)

    if resp.status_code == 200:
        print(f'PDF sent to Telegram: {file_path}')
    else:
        print(f'Failed to send PDF: {resp.status_code} {resp.text}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 send_telegram.py <report.md> [report.pdf]')
        sys.exit(1)

    md_file = sys.argv[1]
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else None

    with open(md_file) as f:
        text = f.read()

    send_telegram_text(text)

    if pdf_file and os.path.exists(pdf_file):
        send_telegram_file(pdf_file, caption=f'Daily Report {os.path.basename(md_file)}')
