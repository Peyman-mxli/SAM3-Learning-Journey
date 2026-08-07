# ☁️ Google Colab



# ☁️ Google Colaboratory

### Run Python, AI Models, and Machine Learning Projects Directly in Your Browser.

---

![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![GPU](https://img.shields.io/badge/Free-GPU-success?style=for-the-badge)
![AI Ready](https://img.shields.io/badge/SAM3-Ready-orange?style=for-the-badge)

</div>

---

# 📚 Table of Contents

- What is Google Colab?
- Why Use Google Colab?
- Benefits of Google Colab
- Creating Your First Notebook
- Understanding the Interface
- Running Python Code
- Using GPUs
- Installing Libraries
- Saving Your Work
- Google Drive Integration
- Best Practices
- Summary

---

# 🎯 Learning Objectives

By the end of this chapter, you will be able to:

- ✅ Understand what Google Colab is.
- ✅ Create and manage notebooks.
- ✅ Execute Python code in the cloud.
- ✅ Connect to free GPU hardware.
- ✅ Install Python packages.
- ✅ Save notebooks to Google Drive.
- ✅ Prepare your environment for SAM3.

---

# ☁️ What is Google Colab?

Google Colaboratory (Google Colab) is a free cloud-based development environment provided by Google.

It allows you to write and execute Python code directly in your web browser without installing anything on your computer.

Instead of relying on your local machine, your code runs on Google's cloud infrastructure.

This makes Google Colab an excellent platform for Artificial Intelligence, Machine Learning, and Computer Vision projects.

---

# 🚀 Why Google Colab?

Many AI models require powerful hardware.

Training or running these models on a personal computer can be difficult because they often require:

- High-end GPUs
- Large amounts of RAM
- Complex software installation
- CUDA drivers
- Deep Learning frameworks

Google Colab removes these barriers by providing ready-to-use cloud resources.

You only need:

- A Google account
- An internet connection
- A web browser

That's it.

---

# 🌟 Benefits of Google Colab

Google Colab offers many advantages for AI development.

- 💻 No installation required
- ☁️ Cloud-based execution
- 🚀 Free GPU access
- 🧠 Supports AI and Machine Learning
- 📦 Pre-installed Python libraries
- 💾 Automatic Google Drive integration
- 🤝 Easy notebook sharing
- 🔄 Works on Windows, macOS, and Linux

These features make Google Colab one of the most popular tools for learning Artificial Intelligence.

---

# 🧠 Why We're Using Google Colab

Throughout this learning journey, we'll use Google Colab to run:

- Segment Anything Model 3 (SAM3)
- PyTorch
- Hugging Face Transformers
- OpenCV
- NumPy
- Matplotlib
- Roboflow datasets
- GPU-accelerated Computer Vision projects

Using Google Colab ensures everyone can follow this course without needing expensive hardware.

---

# 💡 Local Computer vs Google Colab

| Local Computer | Google Colab |
|----------------|--------------|
| Requires installation | No installation |
| Uses your hardware | Uses Google's hardware |
| May require a dedicated GPU | Free GPU available |
| Limited by your computer | Cloud resources |
| Manual environment setup | Ready in seconds |

For AI beginners, Google Colab is usually the easiest and fastest option.

---

# 🎯 Key Takeaways

After completing this section, you should understand that:

- ✅ Google Colab is a cloud-based Python environment.
- ✅ It runs entirely in your browser.
- ✅ No local installation is required.
- ✅ Free GPUs are available for AI workloads.
- ✅ It is one of the best platforms for learning SAM3 and Machine Learning.

---

# 📝 Creating Your First Google Colab Notebook

Now that you understand what Google Colab is, it's time to create your first notebook.

A Colab notebook is an interactive document where you can combine:

- Python code
- Text
- Images
- Explanations
- Charts
- AI experiments

This notebook format is extremely common in Artificial Intelligence and Machine Learning.

---

# 🌐 Step 1 — Open Google Colab

Open:

https://colab.research.google.com/

Sign in with your Google account.

Once Colab opens, you can create a new notebook.

---

# ➕ Step 2 — Create a New Notebook

Click:

```text
File → New notebook
```

A new notebook will open in your browser.

The default filename may look similar to:

```text
Untitled0.ipynb
```

---

# 📄 What is an `.ipynb` File?

Google Colab notebooks use the extension:

```text
.ipynb
```

This stands for:

```text
Interactive Python Notebook
```

Unlike a normal `.py` file, a notebook can contain both:

- Executable Python code
- Formatted documentation

This makes notebooks ideal for AI experiments, tutorials, and research.

---

# ✏️ Step 3 — Rename Your Notebook

Click the notebook name at the top-left.

Change:

```text
Untitled0.ipynb
```

to something meaningful.

For example:

```text
SAM3-Colab-Practice.ipynb
```

Using descriptive filenames will make your projects easier to organize later.

---

# 🧩 Understanding Cells

Google Colab notebooks are divided into **cells**.

There are two important types:

## 💻 Code Cells

Code cells contain Python code.

Example:

```python
print("Hello from Google Colab!")
```

Run the cell and you should see:

```text
Hello from Google Colab!
```

---

## 📝 Text Cells

Text cells are used for:

- Titles
- Notes
- Documentation
- Explanations
- Instructions

They support **Markdown** formatting.

Example:

```markdown
# SAM3 Learning Journey

## Google Colab Practice

This notebook is used to learn cloud-based Python development.
```

---

# ▶️ Running a Code Cell

There are several ways to run a cell.

### Method 1 — Play Button

Click the:

```text
▶
```

button beside the code cell.

---

### Method 2 — Keyboard Shortcut

Press:

```text
Shift + Enter
```

This runs the current cell and moves to the next one.

---

### Method 3 — Run Without Moving

Press:

```text
Ctrl + Enter
```

This executes the current cell while keeping it selected.

---

# 🧪 Your First Colab Program

Create a code cell and enter:

```python
print("Welcome to the SAM3 Learning Journey!")
```

Run the cell.

Expected output:

```text
Welcome to the SAM3 Learning Journey!
```

Congratulations! 🎉

You have now executed Python code in the cloud.

---

# 🧠 Try a Simple Python Example

Create another cell:

```python
name = "SAM3"
course = "Computer Vision"

print("Project:", name)
print("Course:", course)
```

Expected output:

```text
Project: SAM3
Course: Computer Vision
```

---

# 🔢 Run a Calculation

Try:

```python
a = 10
b = 5

result = a * b

print(result)
```

Output:

```text
50
```

Even though the code is running in your browser, the actual computation takes place in Google's cloud environment.

---

# 📊 Try a Simple Visualization

Google Colab works extremely well with visualization libraries.

Run:

```python
import matplotlib.pyplot as plt

values = [10, 20, 15, 30, 25]

plt.plot(values)
plt.title("My First Colab Chart")
plt.show()
```

A chart should appear directly underneath the cell.

This ability to combine code and results in one document is one reason notebooks are so useful for AI research.

---

# 🧭 Understanding the Colab Interface

The main Colab interface includes several important areas.

## 📁 Files

The Files panel allows you to:

- Upload files
- View temporary files
- Access datasets
- Manage notebook resources

We'll use this later when working with images and SAM3.

---

## 📚 Table of Contents

The Table of Contents helps organize larger notebooks.

Markdown headings such as:

```markdown
# Introduction
## Installation
## Run Model
```

will automatically create notebook sections.

---

## 🔎 Variables

During execution, Colab can help you inspect variables created inside the notebook.

This is useful when working with:

- Arrays
- Images
- Model outputs
- Machine Learning datasets

---

# 💾 Saving Your Notebook

Colab notebooks are normally saved to Google Drive.

You may see a folder such as:

```text
My Drive/
└── Colab Notebooks/
```

This allows you to access your notebook later from another computer.

---

# 🔄 Notebook Execution Order

One important concept to understand:

Cells can be executed in any order.

For example:

```python
name = "SAM3"
```

and later:

```python
print(name)
```

The second cell only works if the first cell has already been executed.

Because of this, notebook execution order matters.

---

# 🔢 Cell Execution Numbers

When a cell runs, you may see an execution indicator beside it.

This helps you understand which cells have already been executed.

When debugging a notebook, it is often useful to run everything from the beginning.

---

# 🔁 Run All Cells

To execute the notebook from top to bottom, use the Runtime menu.

This is especially useful after:

- Restarting the environment
- Installing packages
- Changing dependencies
- Testing your notebook from a clean state

---

# ⚠️ Important: Colab Sessions Are Temporary

Google Colab does not behave exactly like your personal computer.

The cloud runtime is temporary.

This means files stored only in the runtime can disappear when:

- The session disconnects
- The runtime restarts
- The notebook remains inactive for too long
- A new runtime is assigned

Later we'll learn how to save important files using:

- Google Drive
- GitHub
- Local downloads

---

# 🧪 Mini Practice

Create a notebook named:

```text
Google-Colab-Practice.ipynb
```

Then create the following cells.

### Cell 1 — Title

```markdown
# Google Colab Practice

SAM3 Learning Journey
```

### Cell 2 — Python

```python
print("Google Colab is ready!")
```

### Cell 3 — Variables

```python
platform = "Google Colab"
purpose = "Artificial Intelligence"

print(platform)
print(purpose)
```

### Cell 4 — Calculation

```python
gpu_memory_example = 16
models = 4

print(gpu_memory_example / models)
```

### Cell 5 — Markdown

```markdown
## Next Goal

Learn how to enable GPU acceleration.
```

---

# ✅ Checkpoint

Before continuing, make sure you can:

- ✅ Open Google Colab.
- ✅ Create a notebook.
- ✅ Rename a notebook.
- ✅ Create code cells.
- ✅ Create text cells.
- ✅ Run Python code.
- ✅ Understand notebook execution order.
- ✅ Save your notebook.
- ✅ Understand that Colab runtimes are temporary.

---

# 🚀 Next: GPU Acceleration

Now we reach one of the most important reasons we're using Google Colab:

# ⚡ Connecting to a GPU

In the next section, you'll learn how to:

- Enable GPU acceleration
- Check which GPU Colab assigned
- Verify CUDA
- Check GPU memory
- Test PyTorch
- Understand CPU vs GPU
- Prepare the runtime for SAM3

---

# ⚡ GPU Acceleration in Google Colab

One of the biggest advantages of Google Colab is the ability to use powerful hardware without purchasing an expensive computer.

Many Artificial Intelligence models—including **Segment Anything Model 3 (SAM3)**—perform millions or even billions of mathematical operations.

A normal CPU can execute these operations, but a modern GPU can perform them **much faster** thanks to its massively parallel architecture.

Understanding how to use GPU acceleration is one of the most important skills in modern AI development.

---

# 🖥️ CPU vs GPU

Before enabling a GPU, it's important to understand the difference between a CPU and a GPU.

## CPU (Central Processing Unit)

A CPU is designed to perform many different types of tasks quickly.

It is excellent for:

- Operating systems
- Office applications
- Web browsing
- Programming
- General computing

CPUs typically have:

- Few processing cores
- High clock speeds
- Excellent sequential performance

Think of a CPU as an expert who solves one complex problem at a time.

---

## GPU (Graphics Processing Unit)

A GPU was originally designed to render graphics and video games.

Today, GPUs are also used for:

- Artificial Intelligence
- Deep Learning
- Computer Vision
- Scientific Computing
- Image Processing
- Video Processing

Unlike CPUs, GPUs contain thousands of small processing cores capable of performing many calculations simultaneously.

Think of a GPU as thousands of workers solving many small problems at the same time.

---

# 🤖 Why AI Uses GPUs

Artificial Intelligence models perform enormous amounts of matrix multiplication.

For example:

- Neural Networks
- Transformers
- Stable Diffusion
- YOLO
- Segment Anything Model (SAM3)

These computations are highly parallel, making GPUs dramatically faster than CPUs.

In many cases, a GPU can reduce processing time from hours to minutes.

---

# ☁️ Google Colab Hardware Accelerators

Google Colab provides different hardware options.

Depending on your account and availability, you may receive:

| Hardware | Description |
|----------|-------------|
| CPU | Standard processor |
| T4 GPU | Most common free NVIDIA GPU |
| L4 GPU | Faster than T4 (availability varies) |
| A100 GPU | High-performance GPU (usually Colab Pro) |
| TPU | Google's Tensor Processing Unit for TensorFlow workloads |

For this learning journey, we'll primarily use NVIDIA GPUs.

---

# 🚀 Enable GPU

Open the menu:

```text
Runtime → Change runtime type
```

Under **Hardware accelerator**, select:

```text
GPU
```

Click:

```text
Save
```

Google Colab will restart your runtime using GPU hardware.

---

# ✅ Verify That a GPU Is Connected

Run the following command:

```python
!nvidia-smi
```

You should see information similar to:

```text
+------------------------------------------------------+
| NVIDIA-SMI |
| GPU Name: Tesla T4 |
| Driver Version |
| CUDA Version |
+------------------------------------------------------+
```

If you see an NVIDIA GPU listed, your notebook is successfully connected.

---

# 🔍 Check GPU Using Python

You can also verify the GPU using PyTorch.

```python
import torch

print(torch.cuda.is_available())
```

Expected output:

```text
True
```

If the result is **False**, your notebook is still using the CPU.

---

# 🎮 Display the GPU Name

```python
import torch

print(torch.cuda.get_device_name(0))
```

Example output:

```text
Tesla T4
```

Depending on availability, your GPU may be different.

---

# 📊 Check GPU Memory

```python
!nvidia-smi
```

Look for values similar to:

```text
Memory Usage

0 MiB / 15360 MiB
```

This tells you:

- Total GPU memory
- Currently allocated memory
- Available memory

Knowing your GPU memory is important because larger AI models require more VRAM.

---

# 🔥 CUDA

CUDA is NVIDIA's platform for accelerating computations on GPUs.

Libraries such as:

- PyTorch
- TensorFlow
- OpenCV
- SAM3

use CUDA to execute operations on the GPU instead of the CPU.

Fortunately, Google Colab already includes CUDA, so no manual installation is required.

---

# 🧪 Check CUDA Availability

Run:

```python
import torch

print(torch.version.cuda)
```

Example output:

```text
12.1
```

This indicates the CUDA version currently available in your runtime.

---

# 📦 Check PyTorch Version

```python
import torch

print(torch.__version__)
```

Example:

```text
2.6.0
```

We'll verify compatible versions again when installing SAM3.

---

# 💾 Check System RAM

Run:

```python
!free -h
```

Example output:

```text
RAM

12 GiB
```

This displays the available system memory assigned to your Colab runtime.

---

# 💽 Check Disk Space

Run:

```python
!df -h
```

This command displays:

- Available storage
- Used storage
- Remaining disk space

Large datasets may require several gigabytes of free storage.

---

# ⚠️ Common GPU Issues

## GPU Not Available

Possible reasons:

- Free GPU quota has been exhausted.
- GPU availability is temporarily limited.
- Hardware accelerator is still set to CPU.

---

## torch.cuda.is_available() Returns False

Possible causes:

- GPU runtime not enabled.
- Runtime restarted incorrectly.
- PyTorch installation issue.

---

## Runtime Disconnects

Google Colab sessions may disconnect after long periods of inactivity.

Always save important work to:

- Google Drive
- GitHub

---

# 🧪 Mini Practice

Run the following commands.

```python
!nvidia-smi
```

```python
import torch

print(torch.cuda.is_available())
```

```python
print(torch.cuda.get_device_name(0))
```

```python
print(torch.version.cuda)
```

```python
print(torch.__version__)
```

If everything works correctly, your notebook is fully prepared for GPU-accelerated AI development.

---

# 🎯 Checkpoint

Before continuing, verify that you can:

- ✅ Enable GPU acceleration.
- ✅ Run `nvidia-smi`.
- ✅ Verify CUDA.
- ✅ Detect your GPU using PyTorch.
- ✅ Check GPU memory.
- ✅ Check system RAM.
- ✅ Understand why GPUs are essential for AI.

Congratulations!

Your Google Colab environment is now ready to execute modern Artificial Intelligence models, including **Segment Anything Model 3 (SAM3)**.

---

# 📁 Working with Files in Google Colab

Almost every Artificial Intelligence project requires working with files.

Examples include:

- 📷 Images
- 🎥 Videos
- 📄 Text files
- 📊 CSV datasets
- 🧠 Trained AI models
- 📦 ZIP files
- 📂 Entire datasets

Google Colab makes it easy to upload, organize, and download these files directly from your browser.

---

# 📂 The Colab File System

Every Google Colab session includes a temporary Linux file system.

You can view it by clicking the folder icon on the left side of the screen.

Typical folders include:

```text
/content
```

This is your current working directory.

You can verify it by running:

```python
import os

os.getcwd()
```

Expected output:

```text
/content
```

---

# 📋 List Files

To see everything inside the current folder:

```python
!ls
```

Or with more details:

```python
!ls -lh
```

Example:

```text
sample_data
```

---

# 📁 Create a Folder

Create a new directory:

```python
!mkdir images
```

Verify:

```python
!ls
```

Output:

```text
images
sample_data
```

---

# 📂 Change Directory

```python
%cd images
```

Return:

```python
%cd /content
```

---

# 📤 Upload Files

Google Colab allows you to upload files directly from your computer.

Run:

```python
from google.colab import files

uploaded = files.upload()
```

A file picker will appear.

Select:

- Images
- Videos
- ZIP files
- Datasets

After uploading, the files will appear inside:

```text
/content
```

---

# 🔍 Verify Uploaded Files

```python
!ls
```

Example:

```text
dog.jpg
cat.png
sample_data
```

---

# 🖼️ Display an Image

Suppose you uploaded:

```text
dog.jpg
```

Display it:

```python
from IPython.display import Image

Image("dog.jpg")
```

The image will appear directly below the cell.

This is especially useful when working with Computer Vision models like SAM3.

---

# 📥 Download Files

Suppose you generated:

```text
result.png
```

Download it:

```python
from google.colab import files

files.download("result.png")
```

Your browser will automatically download the file.

---

# 📄 Create a Text File

```python
with open("notes.txt","w") as file:
    file.write("Hello SAM3!")
```

Verify:

```python
!ls
```

---

# 📖 Read a File

```python
with open("notes.txt") as file:
    print(file.read())
```

Output:

```text
Hello SAM3!
```

---

# 📦 Extract ZIP Files

Many AI datasets are distributed as ZIP archives.

Extract them:

```python
!unzip dataset.zip
```

Or into a specific folder:

```python
!unzip dataset.zip -d dataset
```

---

# 🗑️ Delete Files

Delete one file:

```python
!rm notes.txt
```

Delete a folder:

```python
!rm -r images
```

Be careful—deleted files cannot be recovered unless you've saved them elsewhere.

---

# 📊 Check Folder Size

```python
!du -sh .
```

Example:

```text
1.8G
```

Useful for monitoring dataset size.

---

# 💾 Temporary Storage

One of the most important things to remember:

Everything stored in:

```text
/content
```

is **temporary**.

Files may disappear when:

- The runtime disconnects
- You restart the runtime
- Your session expires
- Google assigns you a new virtual machine

Never rely on `/content` for long-term storage.

---

# 📂 Recommended Project Structure

As your AI projects grow, staying organized becomes essential.

Example:

```text
/content/

├── datasets/
│   ├── images/
│   ├── videos/
│   └── labels/
│
├── models/
│
├── outputs/
│
├── notebooks/
│
└── scripts/
```

This structure keeps datasets, trained models, and generated results separated and easy to manage.

---

# 💡 Best Practices

- Keep datasets in their own folder.
- Use descriptive filenames.
- Delete unnecessary files to save disk space.
- Save important work to Google Drive.
- Keep your notebooks organized.

---

# 🧪 Mini Practice

Create a folder:

```python
!mkdir practice
```

Upload an image.

Display it.

Create:

```python
notes.txt
```

Read it.

Download it.

Finally:

Delete it.

Congratulations—you've completed your first file management workflow in Google Colab.

---

# 🎯 Checkpoint

Before continuing, make sure you can:

- ✅ View the Colab file system.
- ✅ Create folders.
- ✅ Upload files.
- ✅ Display images.
- ✅ Create text files.
- ✅ Read files.
- ✅ Download files.
- ✅ Extract ZIP archives.
- ✅ Delete files.
- ✅ Understand temporary storage.

Excellent! 🎉

You're now ready to connect Google Colab to **Google Drive**, where your files can be stored permanently instead
of the temporary Colab environment.

---

# 💾 Google Drive Integration

One of Google Colab's most powerful features is its seamless integration with **Google Drive**.

By connecting your Drive, you can:

- 📂 Store datasets permanently
- 💾 Save notebooks automatically
- 🧠 Save trained AI models
- 📷 Access images and videos
- 🤝 Share files between projects
- 🚀 Continue working from any computer

Unlike the temporary Colab environment, files stored in Google Drive remain available even after your Colab session ends.

---

# ☁️ Why Use Google Drive?

Remember that everything inside:

```text
/content
```

is temporary.

If your runtime disconnects or restarts, anything stored there may be lost.

Google Drive solves this problem by providing permanent cloud storage.

Think of it like this:

```text
Google Colab
│
├── /content
│     Temporary
│
└── Google Drive
      Permanent
```

---

# 🔗 Mount Google Drive

Run the following code:

```python
from google.colab import drive

drive.mount('/content/drive')
```

After executing the cell:

1. Click the authorization link.
2. Sign in with your Google account.
3. Copy the authorization code.
4. Paste it back into Colab.

After a few seconds, your Drive will be connected.

---

# 📂 Verify the Connection

Run:

```python
!ls /content/drive/MyDrive
```

You should see the folders stored in your Google Drive.

Example:

```text
Colab Notebooks
Datasets
Projects
SAM3
```

Congratulations!

Your Google Drive is now mounted successfully.

---

# 📁 Access Your Files

Suppose your Drive contains:

```text
MyDrive/

├── SAM3/
│
├── datasets/
│
└── notebooks/
```

You can navigate using:

```python
%cd /content/drive/MyDrive/SAM3
```

Verify:

```python
!pwd
```

Output:

```text
/content/drive/MyDrive/SAM3
```

---

# 📷 Read an Image from Drive

Suppose:

```text
MyDrive/

└── datasets/
      dog.jpg
```

Display it:

```python
from IPython.display import Image

Image("/content/drive/MyDrive/datasets/dog.jpg")
```

The image will appear directly inside the notebook.

---

# 💾 Save Files to Google Drive

Instead of saving files inside:

```text
/content
```

save them directly to Drive.

Example:

```python
with open("/content/drive/MyDrive/notes.txt","w") as file:
    file.write("SAM3 Learning Journey")
```

The file now exists permanently inside your Drive.

---

# 📦 Save AI Models

During training, models can be several gigabytes in size.

Example:

```python
torch.save(model.state_dict(),
           "/content/drive/MyDrive/models/model.pth")
```

Even if Colab disconnects, your trained model remains safely stored.

---

# 📊 Recommended Folder Structure

A clean project structure makes AI projects much easier to manage.

Example:

```text
MyDrive/

└── AI-Projects/
    │
    ├── datasets/
    │
    ├── notebooks/
    │
    ├── models/
    │
    ├── outputs/
    │
    ├── checkpoints/
    │
    └── documentation/
```

This organization scales well from small experiments to large AI projects.

---

# 🧠 Organizing Multiple Projects

As you work on more AI models, consider creating separate folders.

Example:

```text
AI-Projects/

├── SAM3/
├── YOLO/
├── StableDiffusion/
├── OpenCV/
└── HuggingFace/
```

Each project can contain its own:

- datasets
- notebooks
- trained models
- results
- documentation

---

# 📄 Save a Notebook

Google Colab automatically saves notebooks to Drive.

You can also manually organize them.

Recommended folder:

```text
MyDrive/

└── Colab Notebooks/
```

Or create your own:

```text
MyDrive/

└── SAM3/
      notebooks/
```

---

# 📤 Download from Drive

Files stored in Drive can easily be downloaded.

Navigate to:

```text
drive.google.com
```

Locate your file.

Right-click.

Choose:

```text
Download
```

---

# 🔄 Reconnect Google Drive

If your runtime restarts, simply run:

```python
from google.colab import drive

drive.mount('/content/drive')
```

again.

The connection only lasts for the current Colab session.

---

# ⚠️ Common Issues

## Permission Denied

Usually caused by:

- Wrong folder path
- Drive not mounted
- Authorization expired

Reconnect Drive.

---

## File Not Found

Always verify:

```python
!ls /content/drive/MyDrive
```

before accessing files.

---

## Runtime Restart

A runtime restart disconnects Drive.

Simply mount it again.

---

# 💡 Best Practices

- Store datasets in dedicated folders.
- Save trained models immediately.
- Organize notebooks by project.
- Use meaningful filenames.
- Keep backup copies of important models.
- Delete unnecessary files to save cloud storage.

---

# 🧪 Mini Practice

Create the following structure inside your Google Drive:

```text
AI-Projects/

└── SAM3/

    ├── datasets/
    ├── notebooks/
    ├── models/
    ├── outputs/
    └── checkpoints/
```

Then:

- Mount Google Drive.
- Navigate to the SAM3 folder.
- Create a text file.
- Save it.
- Verify that it appears in Google Drive.
- Disconnect and reconnect your runtime.
- Open the file again.

If everything works correctly, you've successfully created your first permanent AI workspace.

---

# 🎯 Checkpoint

Before continuing, verify that you can:

- ✅ Mount Google Drive.
- ✅ Access folders.
- ✅ Navigate directories.
- ✅ Read files.
- ✅ Save files permanently.
- ✅ Save AI models.
- ✅ Organize projects.
- ✅ Reconnect Drive after restarting.

Excellent! 🎉

You now have a permanent cloud workspace for Artificial Intelligence development.

From this point forward, you'll be able to store datasets, trained models, notebooks, and experiment results safely in Google Drive.

---

# 🚀 Next: Installing Python Libraries

In the next chapter, you'll learn how to:

- Install Python packages
- Upgrade libraries
- Check installed versions
- Resolve dependency conflicts
- Prepare the complete software environment required for Segment Anything Model 3 (SAM3).

---

# 📦 Installing Python Libraries

One of the greatest advantages of Google Colab is that installing Python libraries is quick and simple.

Unlike a local computer, where you may need to configure environments and dependencies manually, Google Colab allows you to install packages with a single command.

This flexibility lets you experiment with the latest AI frameworks in just a few minutes.

---

# 🤔 What is a Python Library?

A Python library is a collection of pre-written code that provides additional functionality.

Instead of writing everything from scratch, developers reuse libraries to perform common tasks.

For example:

- Image processing
- Deep learning
- Data visualization
- Machine learning
- File management
- Scientific computing

Libraries save thousands of hours of development time.

---

# 📥 Installing a Library

The most common command is:

```python
!pip install package_name
```

Example:

```python
!pip install numpy
```

Google Colab will automatically download and install the package.

---

# 📋 Installing Multiple Libraries

You can install several libraries at once.

Example:

```python
!pip install numpy pandas matplotlib
```

This installs:

- NumPy
- Pandas
- Matplotlib

in a single command.

---

# 🔄 Upgrading a Library

Sometimes you need the newest version.

Use:

```python
!pip install --upgrade numpy
```

The existing installation will be updated automatically.

---

# 📌 Installing a Specific Version

Some AI projects require a particular version.

Example:

```python
!pip install torch==2.6.0
```

Installing specific versions helps ensure compatibility with other libraries.

---

# 📚 Check Installed Packages

View installed packages:

```python
!pip list
```

Example:

```text
numpy
torch
opencv-python
matplotlib
transformers
```

---

# 🔍 Check a Package Version

Example:

```python
import torch

print(torch.__version__)
```

Another example:

```python
import numpy

print(numpy.__version__)
```

Knowing package versions is important when reproducing experiments or troubleshooting compatibility issues.

---

# 🧠 Essential Libraries for AI

Throughout this learning journey, you'll frequently use the following libraries.

| Library | Purpose |
|----------|---------|
| NumPy | Numerical computing |
| Pandas | Data analysis |
| Matplotlib | Data visualization |
| OpenCV | Image processing |
| Pillow | Image manipulation |
| PyTorch | Deep Learning |
| Transformers | Hugging Face models |
| Supervision | Computer Vision utilities |
| Roboflow | Dataset management |
| SAM3 | Image & Video Segmentation |

Each of these libraries plays an important role in modern AI workflows.

---

# ⚡ Install AI Libraries

Example:

```python
!pip install torch torchvision torchaudio
```

Install OpenCV:

```python
!pip install opencv-python
```

Install Transformers:

```python
!pip install transformers
```

Install Supervision:

```python
!pip install supervision
```

Later, we'll install the complete set of libraries required by SAM3.

---

# 🔄 Restart the Runtime

Some libraries require a runtime restart after installation.

Google Colab may display:

```text
Restart Runtime
```

Simply click the button.

Or use:

```text
Runtime → Restart session
```

After restarting, rerun your notebook from the beginning.

---

# ⚠️ Dependency Conflicts

Sometimes two libraries require different versions of the same dependency.

If you see messages like:

```text
ERROR:
```

or

```text
Dependency Conflict
```

don't panic.

Possible solutions include:

- Restarting the runtime.
- Installing compatible versions.
- Following the project's official installation instructions.

We'll encounter this occasionally when working with AI frameworks.

---

# 🧪 Verify an Installation

Example:

```python
import cv2

print(cv2.__version__)
```

If no error appears, OpenCV is installed correctly.

Another example:

```python
import matplotlib

print(matplotlib.__version__)
```

Verifying installations helps identify issues early.

---

# 💡 Best Practices

- Install only the libraries you need.
- Restart the runtime when requested.
- Use compatible library versions.
- Verify installations before continuing.
- Follow official installation guides for AI frameworks.

---

# 🧪 Mini Practice

Install:

```python
!pip install numpy pandas matplotlib
```

Then verify each library:

```python
import numpy
import pandas
import matplotlib

print(numpy.__version__)
print(pandas.__version__)
print(matplotlib.__version__)
```

If all versions display correctly, your installation was successful.

---

# 🎯 Checkpoint

Before moving on, make sure you can:

- ✅ Install Python packages.
- ✅ Upgrade packages.
- ✅ Install specific versions.
- ✅ Check installed libraries.
- ✅ Verify package versions.
- ✅ Restart the runtime when necessary.
- ✅ Understand dependency conflicts.

Excellent! 🎉

Your Google Colab environment is now capable of installing and managing the software required for Artificial Intelligence and Computer Vision projects.

---

# 🚀 Next: GitHub Integration

The next section is one of the most important in this course.

You'll learn how to:

- Clone GitHub repositories
- Open notebooks directly from GitHub
- Download projects
- Keep repositories up to date
- Use GitHub as your central hub for AI development

This workflow will be used repeatedly throughout the remainder of the SAM3 Learning Journey.

---

# 🐙 GitHub Integration

GitHub and Google Colab work extremely well together.

Many Artificial Intelligence projects are published on GitHub, including:

- Meta's Segment Anything Model (SAM)
- PyTorch examples
- Hugging Face projects
- OpenCV tutorials
- YOLO implementations
- Research notebooks

Instead of downloading files manually, Google Colab can open notebooks and clone repositories directly from GitHub.

Throughout this learning journey, we'll use GitHub as the primary source for code, notebooks, and project files.

---

# 🤔 Why GitHub?

GitHub allows developers to:

- Store code
- Share projects
- Track changes
- Collaborate with others
- Publish open-source software
- Manage versions
- Download complete repositories

Nearly every modern AI project begins on GitHub.

---

# 🔗 Open a Notebook from GitHub

Google Colab can open notebooks directly from any public GitHub repository.

1. Open Google Colab.
2. Select:

```text
File → Open notebook
```

3. Choose the **GitHub** tab.
4. Paste the repository URL or search by repository name.
5. Select the notebook you want to open.

The notebook will load directly into Colab without downloading it manually.

---

# 📥 Clone a GitHub Repository

The most common way to work with GitHub in Colab is by cloning a repository.

Syntax:

```bash
!git clone <repository-url>
```

Example:

```bash
!git clone https://github.com/Peyman-mxli/SAM3-Learning-Journey.git
```

After cloning, list the files:

```bash
!ls
```

You should see:

```text
SAM3-Learning-Journey
```

---

# 📂 Navigate into the Repository

Move into the project folder:

```bash
%cd SAM3-Learning-Journey
```

Verify your location:

```bash
!pwd
```

Example output:

```text
/content/SAM3-Learning-Journey
```

---

# 📋 View Repository Contents

List all files:

```bash
!ls
```

Detailed view:

```bash
!ls -lah
```

This helps you explore notebooks, datasets, scripts, and documentation.

---

# 🔄 Update a Repository

If the repository receives updates, download the latest changes:

```bash
!git pull
```

Always run this command from inside the repository folder.

---

# 📦 Download a Repository as ZIP

If you don't want to use Git, GitHub also allows you to download a ZIP archive.

1. Open the repository in your browser.
2. Click the green **Code** button.
3. Select:

```text
Download ZIP
```

Upload the ZIP to Colab and extract it:

```bash
!unzip project.zip
```

---

# 🌍 Clone Open-Source AI Projects

Many of the world's leading AI projects are open source.

Examples include:

- Meta
- Hugging Face
- Ultralytics
- OpenCV
- PyTorch

Cloning repositories gives you immediate access to notebooks, examples, and source code.

---

# 🧠 GitHub + Google Drive

A common workflow is:

```text
GitHub
      │
      ▼
Clone Repository
      │
      ▼
Google Colab
      │
      ▼
Run Notebook
      │
      ▼
Save Results
      │
      ▼
Google Drive
```

This workflow combines version control, cloud computing, and permanent storage.

---

# 💡 Recommended Workflow

For every new AI project:

1. Clone the GitHub repository.
2. Open the notebook.
3. Mount Google Drive.
4. Install required libraries.
5. Run the notebook.
6. Save outputs and trained models to Drive.

This workflow keeps your projects organized and reproducible.

---

# ⚠️ Common Issues

## Repository Not Found

Check that:

- The URL is correct.
- The repository is public.
- You have permission to access it.

---

## Folder Already Exists

If the repository has already been cloned:

```bash
!rm -rf SAM3-Learning-Journey
```

Then clone it again:

```bash
!git clone https://github.com/Peyman-mxli/SAM3-Learning-Journey.git
```

---

## Updates Not Appearing

Navigate into the repository and run:

```bash
!git pull
```

If you've made local changes, Git may ask you to resolve conflicts before pulling.

---

# 🧪 Mini Practice

Clone your repository:

```bash
!git clone https://github.com/Peyman-mxli/SAM3-Learning-Journey.git
```

Then:

- Navigate into the project.
- List the files.
- Open the README.
- Explore the folder structure.
- Return to the `/content` directory.

Congratulations! 🎉

You've successfully connected GitHub with Google Colab.

---

# 🎯 Checkpoint

Before continuing, make sure you can:

- ✅ Open GitHub notebooks in Colab.
- ✅ Clone repositories.
- ✅ Navigate project folders.
- ✅ View repository contents.
- ✅ Update repositories using `git pull`.
- ✅ Download ZIP archives.
- ✅ Understand the GitHub → Colab → Drive workflow.

Excellent!

You now know the same workflow used by AI researchers and developers around the world.

---

# 🚀 Next: Preparing the Environment for SAM3

In the next section, we'll verify that your Colab environment is fully ready for the Segment Anything Model 3 (SAM3).

You'll learn how to:

- Verify Python and CUDA versions.
- Confirm GPU availability.
- Check PyTorch compatibility.
- Install the required dependencies.
- Run a complete environment check before starting SAM3.

---

# 🚀 Preparing Your Environment for SAM3

Congratulations!

You've learned how to:

- Use Google Colab
- Enable GPU acceleration
- Work with files
- Connect Google Drive
- Install Python libraries
- Clone GitHub repositories

Before installing **Segment Anything Model 3 (SAM3)**, it's important to verify that your environment is correctly configured.

Think of this section as a **pre-flight checklist** before launching your AI project.

---

# 🎯 Why Verify Your Environment?

Many installation problems can be avoided by checking your environment first.

By the end of this section, you'll know that:

- Python is working correctly.
- Google Colab is connected to a GPU.
- CUDA is available.
- PyTorch is installed correctly.
- Essential AI libraries are working.
- Your notebook is ready for SAM3.

---

# 🐍 Check Python Version

Run:

```python
import sys

print(sys.version)
```

Example output:

```text
3.11.x
```

Don't worry if the version changes slightly over time—Google Colab updates its environment periodically.

---

# 🖥️ Check GPU

Run:

```python
!nvidia-smi
```

You should see information similar to:

```text
Tesla T4
```

or

```text
L4
```

or another NVIDIA GPU assigned by Google.

---

# 🔥 Check CUDA

```python
import torch

print(torch.version.cuda)
```

Example:

```text
12.x
```

CUDA allows PyTorch to use the GPU for accelerated computation.

---

# ⚡ Verify GPU Availability

```python
import torch

print(torch.cuda.is_available())
```

Expected output:

```text
True
```

If the output is `False`, make sure GPU acceleration is enabled:

```text
Runtime → Change runtime type → GPU
```

---

# 🎮 Display the GPU Name

```python
import torch

print(torch.cuda.get_device_name(0))
```

Example output:

```text
Tesla T4
```

Your assigned GPU may differ depending on availability.

---

# 🧠 Check PyTorch

```python
import torch

print(torch.__version__)
```

Example:

```text
2.x.x
```

PyTorch is the deep learning framework used by SAM3.

---

# 📷 Check OpenCV

```python
import cv2

print(cv2.__version__)
```

If a version number appears, OpenCV is installed correctly.

---

# 📊 Check NumPy

```python
import numpy as np

print(np.__version__)
```

NumPy is used for numerical computing and array operations.

---

# 📈 Check Matplotlib

```python
import matplotlib

print(matplotlib.__version__)
```

Matplotlib will be used to visualize images, masks, and model outputs.

---

# 🖼️ Check Pillow

```python
from PIL import Image

print("Pillow is installed!")
```

Pillow is commonly used for loading and manipulating images.

---

# 🌐 Check Internet Connection

Most AI projects download models and datasets from the internet.

Test your connection:

```python
import requests

response = requests.get("https://www.google.com")

print(response.status_code)
```

Expected output:

```text
200
```

---

# 💾 Check Available Disk Space

```python
!df -h
```

Review the available storage before downloading large models or datasets.

---

# 📊 Check System RAM

```python
!free -h
```

This displays the total and available RAM assigned to your Colab session.

---

# 📁 Verify Google Drive

If you've mounted your Drive, verify the connection:

```python
!ls /content/drive/MyDrive
```

You should see your folders listed.

---

# 📦 Verify Git

Run:

```python
!git --version
```

Example:

```text
git version 2.x.x
```

Git is required for cloning repositories throughout this course.

---

# ✅ Environment Checklist

Before continuing, verify the following:

| Component | Status |
|-----------|--------|
| Python | ✅ |
| GPU | ✅ |
| CUDA | ✅ |
| PyTorch | ✅ |
| NumPy | ✅ |
| OpenCV | ✅ |
| Matplotlib | ✅ |
| Pillow | ✅ |
| Internet Connection | ✅ |
| Google Drive | ✅ |
| Git | ✅ |

If every item is working correctly, your environment is fully prepared.

---

# 🧪 Final Verification Script

You can quickly verify the most important components by running:

```python
import torch
import cv2
import numpy as np
import matplotlib
from PIL import Image
import sys

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Pillow: Installed")

print("\n✅ Environment Ready for SAM3!")
```

If the script completes without errors, your setup is ready for the next stage.

---

# 🎯 Checkpoint

Before moving on, make sure you can:

- ✅ Verify your Python installation.
- ✅ Confirm GPU access.
- ✅ Check CUDA support.
- ✅ Verify PyTorch.
- ✅ Verify OpenCV.
- ✅ Verify NumPy.
- ✅ Verify Matplotlib.
- ✅ Verify Pillow.
- ✅ Confirm internet connectivity.
- ✅ Access Google Drive.
- ✅ Run the final verification script successfully.

Congratulations! 🎉

Your Google Colab environment is now fully prepared for AI development.

You are ready to begin working with real AI models.

---

# 🚀 Next Chapter

# 🤗 06 — Hugging Face

In the next chapter, you'll learn how to:

- Create a Hugging Face account.
- Explore AI models and datasets.
- Generate Access Tokens.
- Download models.
- Use Hugging Face Hub.
- Prepare everything needed for Segment Anything Model 3 (SAM3).

Let's continue building your AI toolkit!

---

# ⚡ Google Colab Productivity Tips

As your AI projects become larger and more complex, learning how to work efficiently in Google Colab will save you a significant amount of time.

Professional AI engineers rely on keyboard shortcuts, organized notebooks, and efficient workflows to improve productivity and reduce mistakes.

This section introduces practical tips that will help you work faster and more effectively throughout the remainder of this learning journey.

---

# ⌨️ Essential Keyboard Shortcuts

Learning a few keyboard shortcuts can dramatically improve your workflow.

| Shortcut | Action |
|-----------|--------|
| **Shift + Enter** | Run the current cell and move to the next |
| **Ctrl + Enter** | Run the current cell without moving |
| **Alt + Enter** | Run the current cell and insert a new cell below |
| **Ctrl + M A** | Insert a code cell above |
| **Ctrl + M B** | Insert a code cell below |
| **Ctrl + M D** | Delete the current cell |
| **Ctrl + M M** | Convert the selected cell to Markdown |
| **Ctrl + M Y** | Convert the selected cell to Code |

Learning these shortcuts can greatly speed up notebook development.

---

# 📝 Write Better Markdown

Well-organized notebooks are easier to understand and maintain.

Use Markdown to separate your notebook into logical sections.

Example:

```markdown
# Project Title

## Installation

## Import Libraries

## Load Dataset

## Train Model

## Results

## Conclusion
```

A clean notebook is easier to debug, share, and revisit later.

---

# 📚 Use a Table of Contents

Large notebooks can become difficult to navigate.

Google Colab automatically generates a Table of Contents based on your Markdown headings.

Example:

```markdown
# Introduction

## Installation

## Dataset

## Model

## Evaluation
```

This allows you to jump directly to any section of the notebook.

---

# 💬 Comment Your Code

Adding comments makes your notebooks easier to understand.

Example:

```python
# Load the image
image = cv2.imread("cat.jpg")

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

Future you—and anyone reading your notebook—will appreciate the extra context.

---

# 🔄 Restarting the Runtime

Sometimes your notebook may stop behaving as expected.

Common reasons include:

- Installing new libraries
- Memory errors
- Dependency conflicts

To restart the environment:

```text
Runtime → Restart session
```

After restarting, remember to run your notebook again from the beginning.

---

# 🧹 Clear Unused Variables

Large AI models consume significant amounts of RAM.

Instead of creating many unnecessary variables, reuse existing ones whenever possible.

Restarting the runtime occasionally also helps free memory.

---

# 📁 Organize Your Projects

Avoid storing everything in one folder.

Recommended structure:

```text
AI-Projects/

├── notebooks/
├── datasets/
├── models/
├── outputs/
├── checkpoints/
└── documentation/
```

Keeping projects organized makes collaboration and maintenance much easier.

---

# 💾 Save Frequently

Although Google Colab automatically saves notebooks, it's still a good habit to save your work regularly.

You can also:

- Save a copy to Google Drive.
- Download the notebook as `.ipynb`.
- Export the notebook as a Python script.

Never assume your session will stay active forever.

---

# 📤 Download Your Notebook

To keep a local backup:

```text
File
    ↓
Download
    ↓
Download .ipynb
```

Or:

```text
Download .py
```

Having local backups is always recommended for important projects.

---

# 🤝 Share Your Notebook

Google Colab notebooks can be shared just like Google Docs.

Click:

```text
Share
```

You can then:

- Allow others to view your notebook.
- Allow collaborators to edit.
- Generate a shareable link.

This is especially useful when working on research or team projects.

---

# 🚀 Use GPU Only When Needed

GPU resources are limited.

If you're only writing documentation or editing Markdown, switch back to the CPU runtime.

Enable the GPU only when running AI models or computationally intensive tasks.

This helps conserve your available GPU quota.

---

# ⚠️ Avoid Running Cells Out of Order

One of the most common mistakes beginners make is executing cells randomly.

Example:

```python
print(model)
```

before creating:

```python
model = MyModel()
```

This results in an error because the variable doesn't exist yet.

As a best practice, execute notebooks from top to bottom.

---

# 📦 Install Libraries Together

Instead of installing one package at a time:

```python
!pip install numpy
!pip install pandas
!pip install matplotlib
```

Install them together:

```python
!pip install numpy pandas matplotlib
```

This reduces installation time and simplifies your notebook.

---

# 🧠 Use Descriptive Notebook Names

Avoid names like:

```text
Untitled1.ipynb
```

Instead, use meaningful names:

```text
SAM3-Image-Segmentation.ipynb
```

Good naming conventions make projects much easier to locate later.

---

# 🛡️ Protect Your Work

Remember:

- Files inside `/content` are temporary.
- Save important work to Google Drive.
- Push valuable projects to GitHub.
- Keep backups of trained models.

Never rely on temporary storage for important files.

---

# ❌ Common Beginner Mistakes

Avoid these common issues:

- Forgetting to mount Google Drive.
- Running notebook cells in the wrong order.
- Ignoring installation error messages.
- Forgetting to restart the runtime after installing packages.
- Saving important files only inside `/content`.
- Using confusing filenames.

Learning to avoid these mistakes will save you hours of troubleshooting.

---

# 💡 Pro Tips

- Keep one project per notebook whenever possible.
- Use Markdown to document your work.
- Organize datasets into dedicated folders.
- Restart the runtime when performance degrades.
- Keep your GitHub repositories updated.
- Save checkpoints during long-running experiments.
- Verify your environment before training AI models.

These habits will make your workflow cleaner, faster, and more professional.

---

# 🎯 Checkpoint

By now, you should be able to:

- ✅ Use essential keyboard shortcuts.
- ✅ Organize notebooks effectively.
- ✅ Write clear Markdown documentation.
- ✅ Restart the runtime when necessary.
- ✅ Save and back up your work.
- ✅ Share notebooks with collaborators.
- ✅ Avoid common beginner mistakes.
- ✅ Follow professional Google Colab workflows.

Excellent! 🎉

You're now using Google Colab more like an experienced AI developer than a beginner.

---

# 📚 Chapter Summary

Congratulations! 🎉

You have successfully completed the **Google Colab** chapter.

At this point, you have everything you need to begin developing Artificial Intelligence projects in a cloud-based environment.

Google Colab is one of the most widely used platforms for AI research, education, and rapid prototyping. Throughout the rest of this learning journey, it will become your primary development environment for building and experimenting with modern AI models.

---

# 📝 What You Learned

During this chapter, you learned how to:

- ✅ Understand what Google Colab is.
- ✅ Create and manage notebooks.
- ✅ Execute Python code in the cloud.
- ✅ Work with Markdown and code cells.
- ✅ Enable GPU acceleration.
- ✅ Understand the difference between CPUs and GPUs.
- ✅ Verify CUDA and GPU availability.
- ✅ Upload, download, and organize files.
- ✅ Work with Google Drive.
- ✅ Install and manage Python libraries.
- ✅ Clone GitHub repositories.
- ✅ Prepare your development environment for SAM3.
- ✅ Improve your workflow using Google Colab productivity tips.

These are the same fundamental skills used daily by AI engineers, data scientists, researchers, and machine learning practitioners.

---

# 🎯 Skills Acquired

After completing this chapter, you can confidently:

- Create professional Google Colab notebooks.
- Execute Python programs in the cloud.
- Connect to NVIDIA GPUs.
- Install AI frameworks and dependencies.
- Manage files and datasets.
- Store projects permanently using Google Drive.
- Work directly with GitHub repositories.
- Verify your AI environment before starting a project.
- Organize notebooks using professional best practices.

These skills will be used repeatedly throughout the remainder of this repository.

---

# 📋 Final Checklist

Before continuing, verify that you can complete each of the following tasks.

| Task | Status |
|------|:------:|
| Create a Google Colab notebook | ☐ |
| Rename a notebook | ☐ |
| Execute Python code | ☐ |
| Create Markdown cells | ☐ |
| Enable GPU acceleration | ☐ |
| Verify CUDA | ☐ |
| Check GPU information | ☐ |
| Upload and download files | ☐ |
| Create folders | ☐ |
| Mount Google Drive | ☐ |
| Install Python packages | ☐ |
| Clone a GitHub repository | ☐ |
| Verify the AI environment | ☐ |

If you can complete every item in this checklist, you're ready for the next chapter.

---

# 💡 Key Takeaways

Remember these important concepts:

- Google Colab runs entirely in the cloud.
- GPU acceleration dramatically improves AI performance.
- Files stored in `/content` are temporary.
- Google Drive provides permanent cloud storage.
- GitHub is the preferred way to share and manage AI projects.
- Installing and verifying libraries before starting a project helps prevent many common issues.
- Keeping notebooks organized and well documented makes collaboration easier and projects more maintainable.

These principles will become part of your everyday AI workflow.

---

# 🚀 Preparing for the Next Chapter

Now that your development environment is fully configured, you're ready to explore one of the most important platforms in modern Artificial Intelligence:

# 🤗 Hugging Face

Hugging Face has become the largest open-source hub for AI models, datasets, and machine learning tools.

In the next chapter, you'll learn how to:

- Create a Hugging Face account.
- Explore thousands of AI models.
- Search and download datasets.
- Generate Access Tokens.
- Use the Hugging Face Hub.
- Connect Hugging Face with Google Colab.
- Prepare your account for Segment Anything Model 3 (SAM3).

By the end of the next chapter, you'll know how to access and use many of the same AI resources trusted by researchers and developers around the world.

---

# 🌟 What's Coming Next?

The next chapters in the **SAM3 Learning Journey** are:

```text
06 - Hugging Face
07 - Roboflow
08 - NVIDIA T4
09 - Install Libraries
10 - Download SAM3
11 - Run Your First Inference
12 - Image Segmentation
13 - Video Segmentation
14 - Fine-Tuning SAM3
15 - Real-World Projects
```

Each chapter builds on the previous one, gradually taking you from setting up your environment to running and customizing advanced AI models.

---

# 🎉 Congratulations!

You have successfully completed the **Google Colab** chapter.

Your cloud-based AI development environment is now fully configured and ready.

Take a moment to review the checklist, ensure everything is working correctly, and then continue to the next chapter.

#  Next Chapter

# **06 — Hugging Face**

Let's continue your journey into modern Artificial Intelligence!

---

# 👨‍💻 Author

**Peyman Miyandashti**

🎓 Information Technology and Digital Innovation Engineering  
🏫 Universidad Politécnica de Baja California (UPBC)

Passionate about:

- 🤖 Artificial Intelligence
- 👁️ Computer Vision
- 🧠 Machine Learning
- 🔒 Cybersecurity
- 💻 Software Development

## 🌐 Connect with Me

- **GitHub:** https://github.com/Peyman-mxli
- **LinkedIn:** *(Add your LinkedIn profile if you'd like)*
- **YouTube:** *(Add your YouTube channel if you'd like)*

If this repository helped you learn something new, consider giving it a ⭐ on GitHub. It helps others discover the project and motivates me to continue creating free educational content.

Happy coding! 🚀
