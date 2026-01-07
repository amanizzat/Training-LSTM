"""
CONVERT IMPROVED MODEL TO ONNX
==============================
Converts the improved Keras model to ONNX format for mobile deployment.
"""

import tensorflow as tf
import tf2onnx
import onnx
from pathlib import Path
import os
import sys
import numpy as np

# --- Configuration ---
KERAS_MODEL_PATH = Path('best_model_improved.keras')
ONNX_MODEL_PATH = Path('action_model_improved.onnx')
SEQUENCE_LENGTH = 30
NUM_FEATURES = 258

# --- Custom Objects for Loading ---
# If using custom Attention layer, we need to register it
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class AttentionLayer(Layer):
    """
    Custom Attention Layer (must match training definition)
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        uit = K.tanh(K.dot(x, self.W) + self.b)
        ait = K.dot(uit, K.expand_dims(self.u))
        ait = K.squeeze(ait, -1)
        ait = K.softmax(ait)
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()


def convert_to_onnx():
    print("="*60)
    print("CONVERTING IMPROVED MODEL TO ONNX")
    print("="*60)
    
    # Check if model exists
    if not KERAS_MODEL_PATH.exists():
        # Try alternative name
        alt_path = Path('action_model_improved.keras')
        if alt_path.exists():
            keras_path = alt_path
        else:
            print(f"Error: Model file not found!")
            print(f"  Tried: {KERAS_MODEL_PATH}")
            print(f"  Tried: {alt_path}")
            print("\nPlease run train_model_improved.py first.")
            sys.exit(1)
    else:
        keras_path = KERAS_MODEL_PATH
    
    print(f"\n📂 Loading Keras model from: {keras_path}")
    
    try:
        # Load with custom objects
        custom_objects = {'AttentionLayer': AttentionLayer}
        model = tf.keras.models.load_model(str(keras_path), custom_objects=custom_objects, compile=False)
        print("✓ Model loaded successfully!")
        
    except Exception as e:
        print(f"⚠ Loading with custom objects failed: {e}")
        print("  Trying without custom objects...")
        
        try:
            model = tf.keras.models.load_model(str(keras_path), compile=False)
            print("✓ Model loaded successfully (no custom objects)!")
        except Exception as e2:
            print(f"✗ Failed to load model: {e2}")
            sys.exit(1)
    
    # Print model summary
    print("\n📊 Model Summary:")
    model.summary()
    
    # Define input signature
    input_signature = [
        tf.TensorSpec([None, SEQUENCE_LENGTH, NUM_FEATURES], tf.float32, name='input_layer')
    ]
    
    print(f"\n🔄 Converting to ONNX...")
    print(f"   Input shape: [batch, {SEQUENCE_LENGTH}, {NUM_FEATURES}]")
    
    try:
        # Convert to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(
            model, 
            input_signature=input_signature, 
            opset=13,
            output_path=str(ONNX_MODEL_PATH)
        )
        
        print(f"\n✓ Conversion successful!")
        
    except Exception as e:
        print(f"\n⚠ Standard conversion failed: {e}")
        print("  Trying alternative conversion method...")
        
        try:
            # Alternative: Save as SavedModel first, then convert
            saved_model_path = 'temp_saved_model'
            model.save(saved_model_path, save_format='tf')
            
            import subprocess
            result = subprocess.run([
                'python', '-m', 'tf2onnx.convert',
                '--saved-model', saved_model_path,
                '--output', str(ONNX_MODEL_PATH),
                '--opset', '13'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(result.stderr)
            
            print(f"✓ Alternative conversion successful!")
            
            # Cleanup
            import shutil
            shutil.rmtree(saved_model_path, ignore_errors=True)
            
        except Exception as e2:
            print(f"✗ Alternative conversion also failed: {e2}")
            print("\n💡 Suggestion: Use model version 'v2' (Deep BiLSTM) which has better ONNX compatibility")
            sys.exit(1)
    
    # Verify ONNX model
    print(f"\n🔍 Verifying ONNX model...")
    try:
        onnx_model = onnx.load(str(ONNX_MODEL_PATH))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid!")
    except Exception as e:
        print(f"⚠ ONNX validation warning: {e}")
    
    # Test inference
    print(f"\n🧪 Testing inference...")
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(str(ONNX_MODEL_PATH))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Create dummy input
        dummy_input = np.random.randn(1, SEQUENCE_LENGTH, NUM_FEATURES).astype(np.float32)
        result = session.run([output_name], {input_name: dummy_input})
        
        print(f"✓ Inference test passed!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {result[0].shape}")
        print(f"   Output (probabilities): {result[0][0][:5]}...")  # Show first 5
        
    except Exception as e:
        print(f"⚠ Inference test failed: {e}")
    
    # Print file info
    file_size = ONNX_MODEL_PATH.stat().st_size / 1024
    print(f"\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)
    print(f"📁 Output file: {ONNX_MODEL_PATH}")
    print(f"📦 File size: {file_size:.2f} KB")
    print(f"\n📝 Next steps:")
    print(f"   1. Copy {ONNX_MODEL_PATH} to your Flutter assets folder")
    print(f"   2. Update pubspec.yaml to include the asset")
    print(f"   3. Update your Flutter app to use the new model")


if __name__ == "__main__":
    convert_to_onnx()