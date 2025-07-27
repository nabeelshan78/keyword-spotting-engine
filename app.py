import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tempfile # For handling temporary files safely
import io # For in-memory audio handling

# --- Configuration and Constants ---
# Define Tx and Ty based on your model's input/output expectations
Tx = 5511 # Number of time steps in the input spectrogram
Ty = 1375 # Number of time steps in the output prediction
n_freq = 101 # Number of frequency bins in the spectrogram

# Path to the chime sound file
CHIME_FILE = "chime.wav"

# Ensure output directory exists
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to the architecture diagram image
ARCHITECTURE_DIAGRAM_PATH = "architecture.png"

# --- Helper Functions (from your provided code) ---

def get_wav_info(wav_file_path):
    """Reads a WAV file and returns its rate and data."""
    rate, data = wavfile.read(wav_file_path)
    return rate, data

def graph_spectrogram(wav_file_path):
    """
    Calculates and plots the spectrogram for a WAV audio file.
    Returns the spectrogram data and the matplotlib Figure object.
    """
    rate, data = get_wav_info(wav_file_path)
    nfft = 200  # Window size for the Fast Fourier Transform (FFT).
    fs = 8000   # Sampling frequencies
    noverlap = 120 # Overlap between successive windows

    # Create a new figure for the spectrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Handle mono vs stereo audio
    if data.ndim == 1:
        pxx, freqs, bins, im = ax.specgram(data, NFFT=nfft, Fs=fs, noverlap=noverlap)
    elif data.ndim == 2:
        # If stereo, use the first channel
        pxx, freqs, bins, im = ax.specgram(data[:, 0], NFFT=nfft, Fs=fs, noverlap=noverlap)
    
    ax.set_title("Spectrogram")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    plt.colorbar(im, ax=ax, format="%+2.0f dB").set_label("Amplitude (dB)")
    
    return pxx, fig # Return spectrogram data and the figure

def match_target_amplitude(sound, target_dBFS):
    """Adjusts the sound's amplitude to a target dBFS level."""
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def modelf(input_shape):
    """
    Builds the trigger word detection model using Conv1D and stacked GRUs.

    Arguments:
    input_shape -- tuple, shape of the input (Tx, n_freq)

    Returns:
    model -- Keras Model instance
    """
    # Input layer
    X_input = Input(shape=input_shape)

    # 1D Convolutional layer
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(rate=0.8)(X)

    # First GRU layer
    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(rate=0.8)(X)
    X = BatchNormalization()(X)

    # Second GRU layer
    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(rate=0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(rate=0.8)(X)

    # TimeDistributed Dense layer with sigmoid activation
    X = TimeDistributed(Dense(1, activation='sigmoid'))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X)

    return model

@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_my_model():
    """Loads the pre-trained Keras model weights."""
    try:
        model = modelf(input_shape=(Tx, n_freq))
        model.load_weights('./model/model_weights.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model weights. Make sure 'model_weights.h5' is in the 'model/' directory. Error: {e}")
        return None

# Load the model once
model = load_my_model()

def detect_triggerword(filename_path, model_instance):
    """
    Detects trigger words in an audio file using the loaded model.
    Returns predictions and the matplotlib Figure object for the probability plot.
    """
    # Correct the amplitude of the input file before prediction
    audio_clip = AudioSegment.from_wav(filename_path)
    audio_clip = match_target_amplitude(audio_clip, -20.0)
    
    # Save to a temporary file for spectrogram generation and model input
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
        audio_clip.export(tmp_audio_file.name, format="wav")
        tmp_filename_path = tmp_audio_file.name

    pxx, _ = graph_spectrogram(tmp_filename_path) # Get spectrogram data, ignore figure for now

    # Remove the temporary file after use
    os.unlink(tmp_filename_path)

    # The spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x = pxx.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0) # Add batch dimension

    if model_instance:
        predictions = model_instance.predict(x)
    else:
        st.error("Model not loaded. Cannot make predictions.")
        return None, None

    # Create a new figure for the prediction plot
    fig_pred, ax_pred = plt.subplots(figsize=(10, 3))
    ax_pred.plot(predictions[0, :, 0])
    ax_pred.set_ylabel('Probability')
    ax_pred.set_xlabel('Time Steps')
    ax_pred.set_title('Trigger Word Probability Over Time')
    ax_pred.set_ylim(0, 1) # Ensure y-axis is between 0 and 1
    
    return predictions, fig_pred

def chime_on_activate(filename_path, predictions, threshold):
    """
    Superposes a chime sound onto the audio clip where trigger words are detected.
    Returns the pydub AudioSegment object of the output chimed audio.
    """
    try:
        audio_clip = AudioSegment.from_wav(filename_path)
        chime = AudioSegment.from_wav(CHIME_FILE)
    except FileNotFoundError:
        st.error(f"Chime file '{CHIME_FILE}' not found. Please ensure it's in the same directory as 'app.py'.")
        return None
    except Exception as e:
        st.error(f"Error loading audio files for chiming: {e}")
        return None

    Ty = predictions.shape[1]
    consecutive_timesteps = 0
    i = 0
    
    # Create a copy of the audio_clip to overlay on
    output_audio = audio_clip[:]

    while i < Ty:
        if predictions[0, i, 0] > threshold:
            consecutive_timesteps += 1
            if consecutive_timesteps > 20: # If prediction is higher than the threshold for 20 consecutive output steps
                # Calculate position in milliseconds
                # The model output has Ty steps, and the audio has audio_clip.duration_seconds
                # So, each output step corresponds to (audio_clip.duration_seconds / Ty) seconds
                position_ms = (i / Ty) * audio_clip.duration_seconds * 1000
                output_audio = output_audio.overlay(chime, position=position_ms)
                consecutive_timesteps = 0
                # Skip forward to avoid multiple chimes for a single long detection
                i = 75 * (i // 75 + 1) # This logic is from original notebook, adjust if needed
                continue
        else:
            consecutive_timesteps = 0
        i += 1
    
    # Instead of saving to file and returning path, return the AudioSegment object
    return output_audio

# --- New Function for M4A to WAV Conversion ---
def convert_m4a_to_wav(m4a_filepath, wav_filepath):
    """
    Converts an M4A audio file to a WAV audio file using pydub.
    Requires FFmpeg to be installed and in the system's PATH.
    """
    try:
        audio = AudioSegment.from_file(m4a_filepath, format="m4a")
        audio.export(wav_filepath, format="wav")
        return wav_filepath
    except FileNotFoundError:
        st.error("FFmpeg not found. Please install FFmpeg and ensure it's in your system's PATH for M4A conversion.")
        return None
    except Exception as e:
        st.error(f"Error converting M4A to WAV: {e}")
        return None

# --- Streamlit App Layout ---

st.set_page_config(
    page_title="Trigger Word Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🗣️ Trigger Word Detector")
st.markdown("---")

# --- Sidebar (now only for About) ---
st.sidebar.header("About This App")
st.sidebar.markdown(
    """
    This application demonstrates a trigger word or wake word detection system. 
    It processes an audio input, identifies the presence of a predefined 
    trigger word (e.g., "activate"), and then overlays a "chime" sound 
    at the detected instances.
    """
)

# --- Main Content Area ---

st.header("Upload Audio or Choose an Example")

audio_option = st.radio(
    "Select audio source:",
    ("Upload new audio", "Choose existing audio"),
    index=0 # Default to upload
)

uploaded_file = None
selected_audio_path = None # This will always point to a WAV file

if audio_option == "Upload new audio":
    uploaded_file = st.file_uploader("Upload an audio file (.wav or .m4a)", type=["wav", "m4a"])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Save uploaded file to a temporary location with its original extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_input_file:
            tmp_input_file.write(uploaded_file.getvalue())
            original_uploaded_path = tmp_input_file.name

        if file_extension == "m4a":
            # Create a temporary path for the WAV output
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
                wav_output_path = tmp_wav_file.name
            
            st.info(f"Converting {file_extension} to WAV...")
            converted_path = convert_m4a_to_wav(original_uploaded_path, wav_output_path)
            
            # Clean up the original temporary m4a file after conversion
            os.unlink(original_uploaded_path) 

            if converted_path:
                selected_audio_path = converted_path # Now points to the temporary WAV file
                st.success("File converted to WAV successfully!")
            else:
                st.error("Failed to convert M4A to WAV. Please ensure FFmpeg is installed and accessible.")
                selected_audio_path = None # Indicate failure
        elif file_extension == "wav":
            selected_audio_path = original_uploaded_path # Already a WAV file
            st.success("WAV file uploaded successfully!")
        else:
            st.error("Unsupported file type. Please upload a .wav or .m4a file.")
            # Clean up unsupported file
            os.unlink(original_uploaded_path)
            selected_audio_path = None

elif audio_option == "Choose existing audio":
    # Define some example audio files. Ensure these paths exist or create dummy ones.
    example_audios = {
        "Example 1": "raw_data/dev/1.wav",
        "Example 2": "raw_data/dev/2.wav",
        "Example 3": "raw_data/backgrounds/1.wav",
        "Example 4": "raw_data/backgrounds/2.wav"
    }
    
    # Check if example files exist before offering them
    available_examples = {name: path for name, path in example_audios.items() if os.path.exists(path)}

    if not available_examples:
        st.warning("No example audio files found. Please ensure 'raw_data/dev/1.wav' and 'raw_data/backgrounds/1.wav' exist.")
    else:
        selected_example_name = st.selectbox(
            "Select an example audio:",
            list(available_examples.keys())
        )
        selected_audio_path = available_examples[selected_example_name]

audio_to_process = None

# Only proceed if a valid audio path (which is always WAV now) is available
if selected_audio_path:
    try:
        # AudioSegment.from_wav expects a WAV file, which selected_audio_path now guarantees
        audio_to_process = AudioSegment.from_wav(selected_audio_path) 
        st.subheader("Original Audio")
        st.audio(selected_audio_path, format="audio/wav")

        st.subheader("Spectrogram of Original Audio")
        pxx_data, spectrogram_fig = graph_spectrogram(selected_audio_path)
        st.pyplot(spectrogram_fig)
        plt.close(spectrogram_fig) # Close the figure to free memory

    except FileNotFoundError:
        st.error(f"Selected audio file not found: {selected_audio_path}")
    except Exception as e:
        st.error(f"Error processing selected audio: {e}")

    # Add a slider for the prediction threshold
    threshold = st.slider(
        "Set Prediction Threshold for Chime Overlay",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust this to control the sensitivity of chime activation."
    )

    if st.button("Detect Trigger Word and Generate Chime Audio"):
        if model is None:
            st.error("Model is not loaded. Please check the console for errors.")
        elif audio_to_process is None:
            st.warning("Please select or upload an audio file first.")
        else:
            with st.spinner("Detecting trigger word and generating chimed audio..."):
                try:
                    # Pass the model instance to detect_triggerword
                    predictions, prediction_fig = detect_triggerword(selected_audio_path, model)
                    
                    if predictions is not None and prediction_fig is not None:
                        st.subheader("Prediction Probability Graph")
                        st.pyplot(prediction_fig)
                        plt.close(prediction_fig) # Close the figure

                        # Call chime_on_activate, which now returns an AudioSegment object
                        output_audio_segment = chime_on_activate(selected_audio_path, predictions, threshold)
                        
                        if output_audio_segment:
                            st.subheader("Chimed Output Audio")
                            # Export the AudioSegment to bytes in memory for st.audio
                            buffer = io.BytesIO()
                            output_audio_segment.export(buffer, format="wav")
                            st.audio(buffer.getvalue(), format="audio/wav")
                            st.success("Chimed audio generated successfully!")
                        else:
                            st.error("Failed to generate chimed audio.")
                    else:
                        st.error("Failed to get predictions.")

                except Exception as e:
                    st.error(f"An error occurred during detection or chiming: {e}")

    # Clean up the temporary WAV file created from upload after all processing
    # This ensures that if an M4A was uploaded and converted, its temporary WAV is deleted.
    # If a WAV was uploaded, its temporary WAV is also deleted.
    if audio_option == "Upload new audio" and selected_audio_path and os.path.exists(selected_audio_path):
        try:
            os.unlink(selected_audio_path)
        except Exception as e:
            st.warning(f"Could not delete temporary file {selected_audio_path}: {e}")

st.markdown("---")

# --- How It Works Section (moved to bottom) ---
st.header("How It Works")
st.markdown(
    """
    1.  **Audio Input**: The app takes a `.wav` or `.m4a` audio file as input. 
        If `.m4a`, it's first converted to `.wav` using FFmpeg.
    2.  **Spectrogram Generation**: The audio is converted into a spectrogram, 
        which is a visual representation of the frequencies present in the audio over time. 
        This is what the neural network "sees".
    3.  **Neural Network Prediction**: A 1D Convolutional Neural Network (CNN) 
        followed by Gated Recurrent Units (GRUs) processes the spectrogram. 
        It outputs a probability score for each short segment of the audio, 
        indicating how likely a trigger word is present.
    4.  **Chime Overlay**: If the probability consistently exceeds a certain threshold 
        (configurable via the slider), a chime sound is overlaid onto the original 
        audio at that specific time, indicating a detected trigger word.
    """
)

st.markdown("---")

# --- Model Architecture Section (moved to bottom) ---
st.header("Model Architecture")

# Display the architecture diagram if it exists
if os.path.exists(ARCHITECTURE_DIAGRAM_PATH):
    st.image(ARCHITECTURE_DIAGRAM_PATH, caption='Model Architecture Diagram', use_container_width=True)
else:
    st.warning(f"Architecture diagram '{ARCHITECTURE_DIAGRAM_PATH}' not found. Please place it in the same directory as 'app.py'.")

st.markdown(
    """
    The model is a sequential Keras model designed for sequence-to-sequence prediction:

    * **Input Layer**: Takes the spectrogram of shape `(Tx, n_freq)`.
    * **Conv1D Layer**: 
        * `filters=196`, `kernel_size=15`, `strides=4`. 
        * Extracts local features from the spectrogram.
    * **Batch Normalization**: Stabilizes and accelerates training.
    * **ReLU Activation**: Introduces non-linearity.
    * **Dropout (0.8)**: Prevents overfitting by randomly setting a fraction of input units to 0 at each update during training time.
    * **GRU Layer 1**: 
        * `units=128`, `return_sequences=True`. 
        * Processes the sequential data and passes its output to the next layer. `return_sequences=True` is crucial for stacked RNNs.
    * **Dropout (0.8)**: Another dropout layer.
    * **Batch Normalization**: Again for stability.
    * **GRU Layer 2**: 
        * `units=128`, `return_sequences=True`. 
        * Further processes the sequential information.
    * **Dropout (0.8)**: Final dropout layer.
    * **Batch Normalization**: Final batch normalization.
    * **TimeDistributed Dense Layer**: 
        * `Dense(1, activation='sigmoid')`. 
        * Applies a Dense layer to *each* timestep of the GRU output, collapsing the 128 units into a single probability score (between 0 and 1) per timestep.
    """
)

st.markdown("---")
st.info("Developed by an AI/ML Engineer with a passion for building intelligent systems.")