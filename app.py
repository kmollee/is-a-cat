from fastai.vision.all import *
import gradio as gr


# setup predict
def is_cat(x):
    return x[0].isupper()


def predict(img):
    img = PILImage.create(img).resize((192, 192))
    pred, pred_idx, probs = learn.predict(img)
    # print(pred,pred_idx,probs)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


MODEL_PATH = "./model.pkl"
IMAGE_DIR = "images"

# load exampels files
example_images = []
for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith(".jpg"):
            example_images.append(os.path.join(root, file))


# local model path
learn = load_learner(MODEL_PATH)
# category
labels = learn.dls.vocab

gr_interface = gr.Interface(
    title="Is it a cat?",
    description="Cat vs Dog Classifier",
    fn=predict,
    inputs="image",
    outputs="label",
    interpretation="default",
    examples=example_images,
)
gr_interface.launch()
