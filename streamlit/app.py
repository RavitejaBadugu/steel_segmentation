import streamlit as st
import numpy as np
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests

def process(image_file):
    encoder=MultipartEncoder(fields={'file':('filename', image_file, 'image/png')})
    response=requests.post('http://fastapi:8000/segmentation',data=encoder,
                  headers={'Content-Type':encoder.content_type})
    return response.json()  

st.title('welcome to segmentation app')

st.header('please upload the picture of steel below to get segmentaions')

st.subheader('''
             Image can have no mask, one mask or multiple masks can be possible. It is trained on 4 classes.
             ''')

file=st.file_uploader('upload image',type=['png','jpg'])

def rle_2_mask(encoded,size=[256,1600]):
    enc_list=encoded.split()
    mask=np.zeros(shape=(size[0]*size[1],))
    height=size[1]
    width=size[0]
    pixels=enc_list[::2]
    counts=enc_list[1::2]
    for i,p in enumerate(pixels):
        assert int(p)-1>=0
        start=int(p)-1
        for j in range(int(counts[i])):
            mask[start+j]=1
    return np.flipud( np.rot90(mask.reshape(height,width),k=1 ))


if file:
    response=process(file)
    st.write('input image is')
    st.image(file)
    encoded_masks=list(response.values())
    for i in range(4):
        st.write(f"for class {i}")
        mask=rle_2_mask(encoded_masks[i])
        st.image(mask)


    