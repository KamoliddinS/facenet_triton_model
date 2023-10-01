
# FaceNet Model for Face Recognition on Nvidia Triton Server

## Introduction

This repository contains a FaceNet model for face recognition that can be deployed on Nvidia Triton Server. FaceNet is a deep learning model for face recognition that generates embeddings for faces, making it suitable for various face-related tasks.

## Model Files

1. `model.savedmodel`: This directory contains the FaceNet model in the SavedModel format. You can use this model for inference with Nvidia Triton Server.

2. `config.pbtxt`: This configuration file is required to set up the model on Triton Server. It specifies important details like the model name, version, input and output tensor names, and more.

## Getting Started

To use the FaceNet model with Nvidia Triton Server, follow these steps:

1. **Install Nvidia Triton Server**: If you haven't already, you can download and install Nvidia Triton Server from the official Nvidia website.

2. **Clone this Repository**:

   ```shell
   git clone https://github.com/KamoliddinS/facenet_triton_model.git
   cd facenet_triton_model
   ```

3. **Deploy the Model**:

   - Copy the `model.savedmodel` directory to the Triton Server model repository directory.
   - Create a Triton Server model repository configuration file (e.g., `my_repo_config.pbtxt`) or edit an existing one. Make sure to specify the model name, version, and other settings based on your requirements. You can refer to the provided `config.pbtxt` for reference.
   - Start Triton Server with the specified model repository configuration:

     ```shell
     tritonserver --model-repository=/path/to/your/model/repository
     ```

4. **Inference**:

   You can now send inference requests to the Triton Server using the Triton Client API or any compatible client. The input and output tensor names and shapes are defined in the `config.pbtxt` file.

## Inference Examples

Here's an example of how to perform inference using the Triton Client API (Python):

```python
import tritonclient.http as tritonhttp

# Define inference parameters
model_name = "facenet"
model_version = "1"
input_data = ...  # Prepare your input data
output_names = ["embeddings"]

# Initialize Triton HTTP client
triton_client = tritonhttp.InferenceServerClient(url="http://localhost:8000")

# Perform inference
response = triton_client.infer(model_name, model_version, input_data, output_names=output_names)

# Retrieve and work with the output embeddings
embeddings = response.as_numpy("embeddings")
print(embeddings)
```

## Additional Resources

- [Nvidia Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)

## License

This FaceNet model and associated files are provided under the [MIT License](LICENSE).

Feel free to reach out if you have any questions or need further assistance!
