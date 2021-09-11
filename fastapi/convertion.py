import tensorflow as tf
import argparse

def dice_coefficent(y_true,y_pred):
    y_true=tf.keras.backend.flatten(tf.cast(y_true,dtype=y_pred.dtype))
    y_pred=tf.keras.backend.flatten(y_pred)
    intersection=tf.keras.backend.sum(y_true*y_pred)
    deno=tf.keras.backend.sum(y_true)+tf.keras.backend.sum(y_pred)+1e-5
    return ((2*intersection)+1e-5)/deno

def bce_dice_loss(y_true,y_pred):
    def get_binary_loss(y_true,y_pred):
        bce=tf.keras.losses.BinaryCrossentropy(reduction='none')
        return bce(y_true,y_pred)
    def get_dice_loss(y_true,y_pred):
        y_true=tf.keras.backend.flatten(tf.cast(y_true,dtype=y_pred.dtype))
        y_pred=tf.keras.backend.flatten(y_pred)
        intersection=tf.keras.backend.sum(y_true*y_pred)
        deno=tf.keras.backend.sum(y_true)+tf.keras.backend.sum(y_pred)+1e-5
        return 1.0-((2*intersection)+1e-5)/deno
    bce_loss=get_binary_loss(y_true,y_pred)
    dice_loss=get_dice_loss(y_true,y_pred)
    total_loss=tf.reduce_mean(bce_loss+dice_loss)
    return total_loss

def load_model(args):
    trained_model=tf.keras.models.load_model(args.model_path,
                                         custom_objects={'bce_dice_loss':bce_dice_loss,'dice_coefficent':dice_coefficent})
                                         
    tf.saved_model.save(trained_model,export_dir=args.export_path)                                         
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str,help='path of keras model')
    parser.add_argument('--export_path',type=str,help='path to save tf formatted model')
    args=parser.parse_args()
    load_model(args)