import requests
import os
from dotenv import load_dotenv

def send_slack_message(webhook_url, message):
    data = {'text': message}
    response = requests.post(webhook_url, json=data)
    if response.status_code != 200:
        raise ValueError(
            'Request to slack returned an error %s, the response is:\n%s'
            % (response.status_code, response.text)
        )



if __name__ == '__main__':
    # webhook_url = 'https://hooks.slack.com/services/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'
    # message = 'Hello, world!'
    # send_slack_message(webhook_url, message)
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    print(webhook_url)

