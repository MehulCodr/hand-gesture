import google.generativeai as genai
from RealtimeSTT import AudioToTextRecorder

global flag

def drone_ghuma_do():
    print("DRONE GHOOM GYA!!!!")

def process_text(text):
    print(text)
    if text == "Flip.":
        drone_ghuma_do()
    else:
        response = model.generate_content(contents=f"{text}")
        print(response.text)

if __name__ == '__main__':
    flag = False
    genai.configure(api_key="AIzaSyBdpCS6P8xdphp7768Jy6AclyjJpuGcvGs")

    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp",
                              system_instruction="""You are a module which is part of a AI based drone project. Whenever you are asked any question, you are meant to answer it in form only yes or no. There is no need to say anything else. It should either be a yes or a no."""
                              )
    prompt = "Answer the question in only yes or no"

    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder()

    while True:
        recorder.text(process_text)
