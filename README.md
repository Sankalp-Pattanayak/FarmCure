<h1>🌿 Plant Disease Detection System</h1>

<p>Welcome to the <strong>Plant Disease Detection System</strong>! This project leverages deep learning techniques using image classification and CNNs to detect diseases in plant leaves. It highlights the disease-affected areas, sharpens blurry images, and provides a detailed breakdown of healthy vs. diseased parts of the leaf. The application also suggests relevant treatment articles and YouTube videos for further guidance.</p>
<h2>Dataset Link</h2>
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data
<h2>🛠 Features</h2>
<ul>
  <li><strong>Disease Detection & Highlighting</strong>: The system identifies and visually highlights disease-affected portions of the leaf for easy interpretation.</li>
  <li><strong>Image Sharpening</strong>: Automatically sharpens the image if it's detected as blurry to improve detection accuracy.</li>
  <li><strong>Image Processing</strong>: The app applies various image processing techniques to enhance image quality.</li>
  <li><strong>Percentage Breakdown</strong>: Displays the percentage of healthy vs. diseased areas of the leaf.</li>
  <li><strong>Treatment Suggestions</strong>: Provides relevant articles and YouTube videos based on the detected disease to help users manage plant health.</li>
</ul>

<h2>📋 Installation</h2>
<ol>
  <li>Clone the repository:
    <pre><code>git clone https://github.com/Sankalp-Pattanayak/FarmCure.git</code></pre>
    <pre><code>cd FarmCure</code></pre>
  </li>
  <li>Install the required packages:
    <pre><code>pip install -r requirement.txt</code></pre>
  </li>
</ol>

<h2>🚀 Usage</h2>
<p>To run the application, execute the <code>main.py</code> file:</p>
<pre><code>python main.py</code></pre>

<h2>🎯 Functionality Overview</h2>
<ol>
  <li><strong>Image Input</strong>: Users upload an image of a plant leaf.</li>
  <li><strong>Image Processing</strong>: The image undergoes sharpening (if blurred) and segmentation to highlight disease-affected areas.</li>
  <li><strong>Disease Detection</strong>: The app detects whether the plant is diseased and calculates the percentage of healthy vs. diseased parts.</li>
  <li><strong>Treatment Suggestions</strong>: If a disease is detected, relevant articles and YouTube videos are displayed to assist in treatment.</li>
  <li><strong>Result Display</strong>: A visual comparison of the original and processed images, along with the detection results, is shown.</li>
</ol>

<h2>📄 License</h2>
<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
