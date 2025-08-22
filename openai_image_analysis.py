# pip install openai>=1.40  (or your current latest)
from openai import OpenAI
import base64, mimetypes

client = OpenAI()

def encode_image(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    b64 = base64.b64encode(open(path, "rb").read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def cxr_demo_classify(image_path):
    img_data_url = encode_image(image_path)

    system_prompt = (
        "You are a medical education assistant. "
        "For the given chest X-ray, provide a NON-DIAGNOSTIC summary. "
        "Return JSON with keys: category in {likely_normal, likely_pneumonia, unsure}, "
        "confidence in [0,1], and rationale (one sentence). "
        "If the image is not a chest X-ray or quality is poor, return category='unsure'. "
        "Do NOT give medical advice or diagnosis."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",           # or "gpt-4o"
        response_format={"type":"json_object"},  # ask for JSON back
        messages=[{
            "role": "system", "content": system_prompt
        },{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image for educational purposes only."},
                {"type": "image_url", "image_url": {"url": img_data_url}}
            ],
        }],
        temperature=0.2,
        max_tokens=300,
    )

    print(resp.choices[0].message.content)  # JSON string

cxr_demo_classify("/path_to_image/person2_bacteria_3.jpeg")
