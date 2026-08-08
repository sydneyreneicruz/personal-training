from fastai.vision.all import *
import gradio as gr

learn = load_learner('model.pkl')
categories = ('water pokemon', 'fire pokemon', 'psychic pokemon')
def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories,map(float,probs)))

image = gr.Image(type="numpy")
label = gr.Label()
examples = ['test_4.jpg', 'test_5.jpg']

if __name__ == "__main__":
    intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
    intf.launch(inline=False,server_name="0.0.0.0", server_port=7860)