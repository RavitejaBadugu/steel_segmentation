from PIL import Image
import io
import requests
import json
import numpy as np

def load_image(image_bytes):
    img=Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img=img.resize((1600,256))
    img=np.array(img)/255.0
    patches=np.zeros((7,256,416,3))
    for id_,j in enumerate(range(0,(1600-200),200)):
        Y_2=416
        y_1=j
        y_2=j+416
        if y_2>1600:
            y_2=1600
            Y_2=400
        patches[id_,:,:Y_2,:]=img[:,y_1:y_2,:]
    return patches

def load_model(URL,instances):
    data=json.dumps({"signature_name": "serving_default",'instances':instances.tolist()})
    headers={"content-type": "application/json"}
    p=requests.post(URL,data,headers)
    predictions=json.loads(p.text)['predictions']
    return np.asarray(predictions)


def combined_masks(pred_masks):
    final_mask=np.zeros((256,1616,4))
    for i,off1 in enumerate(range(0,(1600-200),200)):
        if i==0:
            final_mask[:,0:416,:]=pred_masks[i,]
            prev_idx=416
        else:
            pre_idx=off1
            final_mask[:,pre_idx:prev_idx,:]=(pred_masks[i-1,:,200:416,:] \
                                                +pred_masks[i,:,:216,:])/2
            final_mask[:,prev_idx:off1+416,:]=pred_masks[i,:,216:,:]
            prev_idx=off1+416
    return final_mask[:,:1600,:]

def mask_2_rle(name,mask,threshold=0.5):
    mask= np.asarray(mask)
    assert mask.shape[-1]==4
    rle={}
    for c in range(4):
        curr_mask=mask[:,:,c].T.flatten()
        curr_mask=np.where(curr_mask>threshold,1,0)
        zero_cupy=np.asarray([0])
        curr_mask=np.concatenate([zero_cupy,curr_mask,zero_cupy])
        curr_mask=np.where(curr_mask[1:]!=curr_mask[:-1])[0]+1
        curr_mask[1::2]-=curr_mask[::2]
        rle[f"{name}_{c+1}"]=' '.join(str(x) for x in curr_mask)
    return rle


def get_prediction(image_bytes):
    patches=load_image(image_bytes)
    predictions=load_model('http://tf_serving:8501/v1/models/segmentation_model:predict',patches)
    masks=combined_masks(predictions)
    rle=mask_2_rle('temp_image',masks,0.1)
    return rle