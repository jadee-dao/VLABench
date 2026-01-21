from VLABench.evaluation.model.vlm.base import *

class Gemini(BaseVLM):
    def __init__(self, api_key=None, base_url=None, model="gemini-robotics-er-1.5-preview") -> None:
        self.api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY") or api_key
        self.base_url = base_url
        self.model = model
        super().__init__()

    def evaluate(self, input_dict, language, with_CoT=False):
        from google import genai
        from google.genai import types
        
        ti_list = []
        ti_list.append(["text", input_dict["pre_prompt"] ])
        ti_list = get_ti_list(input_dict, language, with_CoT)

        client = genai.Client(api_key=self.api_key)
        
        contents = []
        for item in ti_list:
            if item[0] == "text":
                contents.append(item[1])
            elif item[0] == "image":
                image_path = item[1]
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                
                # Determine mime type based on extension, default to png
                mime_type = 'image/jpeg' if image_path.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
                
                contents.append(types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ))

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0, # Set to 0 for deterministic output as usually desired in eval
                )
            )
            content = response.text
        except Exception as e:
            print(f"Error querying Gemini: {e}")
            return {"format_error": str(e)}

        output = {}
        output["origin_output"] = content
        try:
            # Try to find JSON block first
            if "```json" in content:
                json_data = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                 json_data = content.split("```")[1]
            else:
                json_data = content
                
            output["skill_sequence"] = json.loads(json_data)
        except:
            output["format_error"] = "format_error"
        return output

    def get_name(self):
        return "Gemini"