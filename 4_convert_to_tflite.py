"""
CONVERT KERAS MODEL TO TFLITE
=============================
Converts your Keras model to TensorFlow Lite format for on-device inference.

TFLite advantages:
- Runs directly on Android/iOS
- No server needed
- Fast inference (~10-50ms)
- Small model size

Usage:
    python convert_to_tflite.py
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

# --- Configuration ---
KERAS_MODEL_PATH = Path('best_model.keras')  # Your trained model
TFLITE_MODEL_PATH = Path('action_model.tflite')
LABELS_PATH = Path('action_labels.npy')
LABELS_TXT_PATH = Path('action_labels.txt')

# For improved model
KERAS_MODEL_IMPROVED_PATH = Path('best_model_improved.keras')
TFLITE_MODEL_IMPROVED_PATH = Path('action_model_improved.tflite')

# For single-frame model
KERAS_MODEL_SINGLE_PATH = Path('best_model_single_frame.keras')
TFLITE_MODEL_SINGLE_PATH = Path('action_model_single_frame.tflite')
# ---------------------


def convert_model(keras_path, tflite_path, model_name="Model"):
    """Convert a Keras model to TFLite"""
    
    print(f"\n{'='*60}")
    print(f"Converting: {model_name}")
    print(f"{'='*60}")
    
    if not keras_path.exists():
        print(f"⚠ Model not found: {keras_path}")
        return False
    
    print(f"📂 Loading Keras model from: {keras_path}")
    
    try:
        # Load model (try with and without compile)
        try:
            model = tf.keras.models.load_model(str(keras_path), compile=False)
        except Exception as e:
            print(f"  Trying with custom objects...")
            # For models with custom layers
            from tensorflow.keras.layers import Layer
            from tensorflow.keras import backend as K
            
            class AttentionLayer(Layer):
                def __init__(self, **kwargs):
                    super(AttentionLayer, self).__init__(**kwargs)
                
                def build(self, input_shape):
                    self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
                    self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
                    self.u = self.add_weight(name='attention_context', shape=(input_shape[-1],), initializer='glorot_uniform', trainable=True)
                    super(AttentionLayer, self).build(input_shape)
                
                def call(self, x):
                    uit = K.tanh(K.dot(x, self.W) + self.b)
                    ait = K.dot(uit, K.expand_dims(self.u))
                    ait = K.squeeze(ait, -1)
                    ait = K.softmax(ait)
                    ait = K.expand_dims(ait)
                    return K.sum(x * ait, axis=1)
                
                def compute_output_shape(self, input_shape):
                    return (input_shape[0], input_shape[-1])
                
                def get_config(self):
                    return super(AttentionLayer, self).get_config()
            
            custom_objects = {'AttentionLayer': AttentionLayer}
            model = tf.keras.models.load_model(str(keras_path), custom_objects=custom_objects, compile=False)
        
        print("✓ Model loaded successfully!")
        
        # Print model info
        print(f"\n📊 Model Summary:")
        input_shape = model.input_shape
        output_shape = model.output_shape
        print(f"   Input shape: {input_shape}")
        print(f"   Output shape: {output_shape}")
        
        # Convert to TFLite
        print(f"\n🔄 Converting to TFLite...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimization options
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # For better compatibility
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        print(f"💾 Saving TFLite model to: {tflite_path}")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Print file size
        file_size_kb = tflite_path.stat().st_size / 1024
        file_size_mb = file_size_kb / 1024
        
        print(f"\n✓ Conversion successful!")
        print(f"   File size: {file_size_kb:.2f} KB ({file_size_mb:.2f} MB)")
        
        # Verify the model
        print(f"\n🔍 Verifying TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   Input: {input_details[0]['shape']} ({input_details[0]['dtype']})")
        print(f"   Output: {output_details[0]['shape']} ({output_details[0]['dtype']})")
        
        # Test inference
        print(f"\n🧪 Testing inference...")
        input_shape = input_details[0]['shape']
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"   Test output shape: {output.shape}")
        print(f"   Test output (first 5): {output[0][:5]}")
        print(f"✓ Model verification passed!")
        
        return True
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_labels_txt():
    """Create a text file version of labels for Flutter"""
    if LABELS_PATH.exists():
        print(f"\n📝 Creating labels text file...")
        labels = np.load(str(LABELS_PATH))
        
        with open(LABELS_TXT_PATH, 'w') as f:
            for label in labels:
                f.write(f"{label}\n")
        
        print(f"✓ Labels saved to: {LABELS_TXT_PATH}")
        print(f"   Labels: {list(labels)}")
    else:
        print(f"⚠ Labels file not found: {LABELS_PATH}")


def main():
    print("\n" + "="*60)
    print("KERAS TO TFLITE CONVERTER")
    print("="*60)
    
    converted_any = False
    
    # Convert original model
    if KERAS_MODEL_PATH.exists():
        if convert_model(KERAS_MODEL_PATH, TFLITE_MODEL_PATH, "Original LSTM Model"):
            converted_any = True
    
    # Convert improved model
    if KERAS_MODEL_IMPROVED_PATH.exists():
        if convert_model(KERAS_MODEL_IMPROVED_PATH, TFLITE_MODEL_IMPROVED_PATH, "Improved BiLSTM Model"):
            converted_any = True
    
    # Convert single-frame model
    if KERAS_MODEL_SINGLE_PATH.exists():
        if convert_model(KERAS_MODEL_SINGLE_PATH, TFLITE_MODEL_SINGLE_PATH, "Single-Frame Model"):
            converted_any = True
    
    # Create labels text file
    create_labels_txt()
    
    # Summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    
    if converted_any:
        print("\n✓ Conversion complete!")
        print("\n📁 Generated files:")
        for f in [TFLITE_MODEL_PATH, TFLITE_MODEL_IMPROVED_PATH, TFLITE_MODEL_SINGLE_PATH]:
            if f.exists():
                print(f"   - {f} ({f.stat().st_size/1024:.1f} KB)")
        if LABELS_TXT_PATH.exists():
            print(f"   - {LABELS_TXT_PATH}")
        
        print("\n📝 Next steps:")
        print("   1. Copy .tflite file to: android/app/src/main/assets/")
        print("   2. Copy action_labels.txt to: assets/")
        print("   3. Add to pubspec.yaml:")
        print("      flutter:")
        print("        assets:")
        print("          - assets/action_model.tflite")
        print("          - assets/action_labels.txt")
        print("   4. Run: flutter pub get")
        print("   5. Rebuild app")
    else:
        print("\n⚠ No models were converted!")
        print("   Make sure you have trained a model first.")
        print("   Run: python train_model.py")


if __name__ == "__main__":
    main()