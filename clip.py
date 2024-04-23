from transformers import CLIPProcessor, CLIPModel
from typing import List
import numpy as np
from PIL import Image
from urllib.request import urlretrieve
from pathlib import Path
# load pre-trained model
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# load preprocessor for model input
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

text_descriptions = ['This is a photo of a Daffodil, The daffodil showcases a large, showy flower with a vibrant yellow trumpet-shaped corona at the center and six white petals that surround it, creating a beautiful contrast.',
 'This is a photo of Snowdrop, The snowdrop displays exquisite, pendulous flowers with three white outer petals and three inner petals that form a bell-like shape, hanging gracefully from a slim stem.',
 'This is a photo of LilyValley, features delicate, bell-shaped flowers with small, white petals that dangle from a slender stalk, accompanied by glossy, dark green lanceolate leaves.',
 'This is a photo of Bluebell, presents a cluster of nodding, tubular flowers that are typically deep blue in color, each with six fused petals forming a bell-shaped structure.',
 'This is a photo of Crocus, reveals slender, goblet-shaped flowers with six petals that bloom in an array of colors, including purple, yellow, white, or striped varieties.',
 'This is a photo of Iris, showcases striking, intricate flowers with three upright petals called standards and three drooping petals called falls, often exhibiting a vibrant mix of colors and patterns.',
 'This is a photo of Tigerlily, boasts large, showy flowers with recurved orange petals covered in dark spots, resembling the coat of a tiger, while displaying prominent stamens and pistils.',
 'This is a photo of Tulip, a small, vibrant yellow flower with delicate, cup-shaped blooms. its smooth, silky petals create a striking contrast against the green foliage. The blooms unfurl to reveal intricate details, with a cluster of yellow-orange stamens at the center.',
 'This is a photo of Fritillary, presents bell-shaped flowers with distinctive patterns, such as checkered or spotted designs, in shades of purple, pink, or white, adding a touch of elegance and allure.',
 'This is a photo of Sunflower, exhibits a large, round flower head with bright yellow petals radiating from a dark brown center disk, creating a sun-like appearance and reaching impressive heights.',
 'This is a photo of Daisy, showcases a classic flower with a yellow center disk surrounded by white or yellow petals, exhibiting a simple and cheerful charm.',
 'This is a photo of Colts foot, displays bright yellow, daisy-like flowers with numerous narrow petals arranged in a dense cluster, emerging before the appearance of its large, hoof-shaped leaves.',
 'This is a photo of Dandelion, a common dandelion flower, characterized by its bright yellow, many-petaled, rosette shape. Each "petal" is actually an individual flower, collectively making up the dandelion\'s composite head. ',
 'This is a photo of Cowslip, features clusters of nodding, bell-shaped yellow flowers with orange or red spots on the inside, held by slender stems amidst a rosette of wrinkled, oval-shaped leaves.',
 'This is a photo of Buttercup, presents glossy, bright yellow flowers with five shiny petals and a cluster of yellow stamens in the center, radiating a joyful and vibrant aura.',
 'This is a photo of Wind flower, delicate, with white petals that radiate from the center. The center of the flower has a cluster of bright yellow stamens, surrounded by a ring of green. The texture of the petals appears to have a subtle veining pattern, contributing to its dainty look.',
 'This is a photo of Pansy, showcases small, velvety flowers with rounded petals in an array of colors, including purple, yellow, blue, or white, often displaying "faces" with dark lines or markings, adding a touch of whimsical charm.']
aggregated_acc = 0
pred_results = {}
for i in range(17): #
    pred_results[i] = []
    acc = 0
    for j in range(1,81):
        num = i*80+j
        if num < 10:
            path = "data/flower/image_000"+str(num)+".jpg"
        elif num >= 10 and num < 100:
            path = "data/flower/image_00"+str(num)+".jpg"
        elif num >= 100 and num < 1000:
            path = "data/flower/image_0"+str(num)+".jpg"
        else:
            path = "data/flower/image_"+str(num)+".jpg"
        sample_path = Path(path)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(sample_path)
        inputs = processor(text=text_descriptions, images=[image], return_tensors="pt", padding=True)
        results = model(**inputs)
        logits_per_image = results['logits_per_image']  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1).detach().numpy()  # we can take the softmax to get the label probabilities
        top_label = np.argsort(-probs)[:min(5, probs.shape[0])][0][0]
        pred_results[i].append(top_label)
        if top_label == i:
            acc+=1
            aggregated_acc+=1
    print("class "+str(i)+": "+str(acc/80))
print("General Accuracy: ", aggregated_acc/1360)