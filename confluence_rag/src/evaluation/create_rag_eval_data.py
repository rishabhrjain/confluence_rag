
import json 
import numpy as np
import time
import json
import boto3

from confluence_rag.config import PAGES_DIR, EVAL_DIR, AWS_PROFILE, AWS_REGION, MODEL_ID

session = boto3.Session(profile_name=AWS_PROFILE)
np.random.seed(50)
bedrock_client = session.client(
        service_name='bedrock-runtime',
        region_name=AWS_REGION  # Replace with your desired region
    )

def get_question_prompt(text, title):

    prompt = f"""
            Generate 5 questions based on the following context. 
            The questions should:
            - Include a mix of factual and thought-provoking questions
            - Focus on key concepts and important details
            - Vary in difficulty (some straightforward, some requiring deeper analysis)
            - Be clearly answerable from the text

            Context:
            Page title: {title}: 
            {text}

            Output only the questions, one per line.
             """
    return prompt.replace('  ', ' ')

def make_llm_request(text):

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "temperature": 0.4, # lower temp means less creative, more focussed
        "top_p": 0.5,
        "top_k": 40,
        "stop_sequences": [],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]
    }

    # Convert the request body to JSON string
    json_body = json.dumps(request_body)

    # Make the API call
    response = bedrock_client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json_body
    )

    # Process the response
    response_body = json.loads(response['body'].read())
    return response_body 

def fetch_questions_for_pages(pages):

    questions = []

    for i, page in enumerate(pages):

        text = get_question_prompt(page['page_content'], page['page_title'])

        llm_response = make_llm_request(text)

        response_text = llm_response['content'][0]['text']
        questions = response_text.split('\n')

        # remove empty strings
        questions = list(filter(None, questions))

        page['questions'] = questions

        questions.append(page)

        time.sleep(2)

    return questions

def main():

    # read clean pages 
    with open(f'{PAGES_DIR}/cleaned_pages.json', 'rb') as f:
        pages = json.load(f)

    
    # randomly sample 100 indices
    random_idx = np.random.randint(low=0, high = len(pages) - 1, size = 100)

    # get pages based on random_idx
    sampled_pages = [pages[idx] for idx in random_idx]

    questions = fetch_questions_for_pages(sampled_pages)


    if not EVAL_DIR.exists():
        EVAL_DIR.mkdir(exist_ok=True, parents=True)

    with open(EVAL_DIR/'rag_eval_test.json', 'w') as f:
        json.dump(questions, f, ensure_ascii=True, indent=4)


if __name__ == "__main__":
    main()