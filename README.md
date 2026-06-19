<p align="center">
  <img src="1 (2).png" width="100%" alt="SHAP vs LIME Banner">
</p>
<br />
<p align="center">
  <a href="#english-version">🇬🇧 English Version</a> │ 
  <a href="#persian-version">🇮🇷 نسخه فارسی</a>
</p>

<hr />

<div id="english-version" lang="en">

<h1 align="center">🔮 SHAP vs LIME: The Ultimate XAI Benchmark Study</h1>
<p align="center">
  <strong>Classification & Regression • 3 Machine Learning Models • Speed & Prediction Stability Analysis</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/XGBoost-Optimized-green.svg" alt="XGBoost">
  <img src="https://img.shields.io/badge/XAI-SHAP%20%26%20LIME-purple.svg" alt="XAI Frameworks">
</p>

<h2>🎯 Overview</h2>
<p>
This framework provides a rigorous, hands-on benchmark comparing the two pillars of Explainable AI (XAI): 
<strong>SHAP (SHapley Additive exPlanations)</strong> and <strong>LIME (Local Interpretable Model-agnostic Explanations)</strong>. 
Instead of looking only at the theory, this project stress-tests both explainers across identical environments, tracking 
execution speed, feature consistency, and local interpretation stability.
</p>

<h2>📊 Key Findings & Experimental Results</h2>
<table width="100%">
  <thead>
    <tr style="background-color: #1f2937; color: white;">
      <th>Task</th>
      <th>Model</th>
      <th>SHAP Time</th>
      <th>LIME Time</th>
      <th>Faster</th>
      <th>Top-3 Agreement</th>
      <th>SHAP Stable?</th>
      <th>LIME Stable?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Classification</b></td>
      <td>Random Forest</td>
      <td>0.21s</td>
      <td>3.98s</td>
      <td style="color: #10b981;"><b>SHAP</b></td>
      <td>78%</td>
      <td>✓ Yes (std=0)</td>
      <td>✗ No (std>0)</td>
    </tr>
    <tr>
      <td><b>Classification</b></td>
      <td>XGBoost</td>
      <td>0.03s</td>
      <td>2.06s</td>
      <td style="color: #10b981;"><b>SHAP</b></td>
      <td>75%</td>
      <td>✓ Yes (std=0)</td>
      <td>✗ No (std>0)</td>
    </tr>
    <tr>
      <td><b>Classification</b></td>
      <td>Logistic Reg.</td>
      <td>0.79s</td>
      <td>2.00s</td>
      <td style="color: #10b981;"><b>SHAP</b></td>
      <td>81%</td>
      <td>✓ Yes (std=0)</td>
      <td>✗ No (std>0)</td>
    </tr>
    <tr>
      <td><b>Regression</b></td>
      <td>Random Forest</td>
      <td>26.28s</td>
      <td>2.31s</td>
      <td style="color: #ef4444;"><b>LIME</b></td>
      <td>72%</td>
      <td>✓ Yes (std=0)</td>
      <td>✗ No (std>0)</td>
    </tr>
    <tr>
      <td><b>Regression</b></td>
      <td>XGBoost</td>
      <td>0.06s</td>
      <td>0.39s</td>
      <td style="color: #10b981;"><b>SHAP</b></td>
      <td>65%</td>
      <td>✓ Yes (std=0)</td>
      <td>✗ No (std>0)</td>
    </tr>
    <tr>
      <td><b>Regression</b></td>
      <td>Linear Reg.</td>
      <td>0.64s</td>
      <td>0.35s</td>
      <td style="color: #ef4444;"><b>LIME</b></td>
      <td>75%</td>
      <td>✓ Yes (std=0)</td>
      <td>✗ No (std>0)</td>
    </tr>
  </tbody>
</table>

<h3>💡 Crucial Insights</h3>
<ul>
  <li><strong>The Stability Paradox:</strong> SHAP is 100% deterministic (Std = 0.0) across consecutive runs on the exact same sample due to its Game Theory background. LIME, being stochastic (sampling-based), introduces variance in feature weights across identical runs.</li>
  <li><strong>The Speed Trade-off:</strong> SHAP’s <code>TreeExplainer</code> is highly optimized for XGBoost (0.03s), but scales poorly on deep Random Forest configurations where <code>KernelExplainer</code> is required, making LIME up to 11x faster in specific regression pipelines.</li>
</ul>

<h2>🚀 Getting Started</h2>
<h3>1. Clone & Setup Environment</h3>
<pre><code>git clone https://github.com/SeyyedSajjadFazeli/xai-shap-vs-lime-benchmark.git
cd xai-shap-vs-lime-benchmark
python -m venv venv
# Windows: venv\Scripts\activate | Unix: source venv/bin/activate</code></pre>

<h3>2. Install Dependencies</h3>
<pre><code>pip install -r requirements.txt</code></pre>

<h3>3. Execute Pipeline</h3>
<pre><code>python shap_vs_lime_comparison.py</code></pre>

</div>

<hr />

<div id="persian-version" lang="fa" dir="rtl">

<h1 align="center">🔮 مقایسه و بنچمارک جامع SHAP vs LIME</h1>
<p align="center">
  <strong>ارزیابی تسک‌های طبقه‌بندی و رگرسیون • بررسی ۳ مدل یادگیری ماشین • تحلیل سرعت محاسبات و پایداری تفاسیر</strong>
</p>

<h2>🎯 مرور کلی پروژه</h2>
<p>
این فریمورک یک مطالعه بنچمارک دقیق و کاملاً پیاده‌سازی شده روی دو ستون اصلی هوش مصنوعی قابل تفسیر (XAI) یعنی 
<strong>SHAP</strong> و <strong>LIME</strong> ارائه می‌دهد. 
هدف این پروژه عبور از تعاریف تئوری و به چالش کشیدن هر دو ابزار در شرایط کاملاً یکسان روی مدل‌های مختلف برای سنجش 
سرعت محاسبات، پایداری خروجی در اجراهای متوالی و میزان توافق الگوریتم‌ها بر سر ویژگی‌های برتر است.
</p>

<h2>📊 یافته‌های کلیدی و نتایج آزمایشات</h2>
<table width="100%" dir="rtl">
  <thead>
    <tr style="background-color: #1f2937; color: white;">
      <th>نوع تسک</th>
      <th>مدل یادگیری ماشین</th>
      <th>زمان اجرای SHAP</th>
      <th>زمان اجرای LIME</th>
      <th>الگوریتم سریع‌تر</th>
      <th>میزان توافق (Top-3)</th>
      <th>پایداری SHAP؟</th>
      <th>پایداری LIME؟</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>طبقه‌بندی (Classification)</b></td>
      <td>Random Forest</td>
      <td>0.21 ثانیه</td>
      <td>3.98 ثانیه</td>
      <td style="color: #10b981;"><b>SHAP</b></td>
      <td>78%</td>
      <td>✓ بله (std=0)</td>
      <td>✗ خیر (std>0)</td>
    </tr>
    <tr>
      <td><b>طبقه‌بندی (Classification)</b></td>
      <td>XGBoost</td>
      <td>0.03 ثانیه</td>
      <td>2.06 ثانیه</td>
      <td style="color: #10b981;"><b>SHAP</b></td>
      <td>75%</td>
      <td>✓ بله (std=0)</td>
      <td>✗ خیر (std>0)</td>
    </tr>
    <tr>
      <td><b>طبقه‌بندی (Classification)</b></td>
      <td>Logistic Reg.</td>
      <td>0.79 ثانیه</td>
      <td>2.00 ثانیه</td>
      <td style="color: #10b981;"><b>SHAP</b></td>
      <td>81%</td>
      <td>✓ بله (std=0)</td>
      <td>✗ خیر (std>0)</td>
    </tr>
    <tr>
      <td><b>رگرسیون (Regression)</b></td>
      <td>Random Forest</td>
      <td>26.28 ثانیه</td>
      <td>2.31 ثانیه</td>
      <td style="color: #ef4444;"><b>LIME</b></td>
      <td>72%</td>
      <td>✓ بله (std=0)</td>
      <td>✗ خیر (std>0)</td>
    </tr>
    <tr>
      <td><b>رگرسیون (Regression)</b></td>
      <td>XGBoost</td>
      <td>0.06 ثانیه</td>
      <td>0.39 ثانیه</td>
      <td style="color: #10b981;"><b>SHAP</b></td>
      <td>65%</td>
      <td>✓ بله (std=0)</td>
      <td>✗ خیر (std>0)</td>
    </tr>
    <tr>
      <td><b>رگرسیون (Regression)</b></td>
      <td>Linear Reg.</td>
      <td>0.64 ثانیه</td>
      <td>0.35 ثانیه</td>
      <td style="color: #ef4444;"><b>LIME</b></td>
      <td>75%</td>
      <td>✓ بله (std=0)</td>
      <td>✗ خیر (std>0)</td>
    </tr>
  </tbody>
</table>

<h3>💡 تحلیل و نکات تجربی مهم</h3>
<ul>
  <li><strong>پارادوکس پایداری (Stability):</strong> الگوریتم SHAP به دلیل پایه نظری قوی در نظریه بازی‌ها، ۱۰۰٪ قطعی (Deterministic) عمل می‌کند و انحراف معیار آن صفر است. اما LIME به دلیل رفتار مبتنی بر نمونه‌برداری تصادفی (Stochastic)، در اجراهای متوالی روی یک نمونه ثابت، وزن‌های متغیری به ویژگی‌ها اختصاص می‌دهد.</li>
  <li><strong>چالش مدیریت زمان (Speed):</strong> ابزار <code>TreeExplainer</code> در SHAP روی مدل XGBoost فوق‌العاده سریع است (0.03 ثانیه)، اما زمانی که مجبور به استفاده از <code>KernelExplainer</code> روی ساختارهای ترکیبی پیچیده (مانند خط‌لوله‌های رگرسیون تصادفی) می‌شود، زمان پردازش به شدت بالا رفته (26.28 ثانیه) و در اینجا LIME بیش از ۱۱ برابر سریع‌تر عمل می‌کند.</li>
</ul>

<h2>🚀 راه‌اندازی و اجرا</h2>
<h3>۱. کلون کردن مخزن و ساخت محیط مجازی</h3>
<pre dir="ltr"><code>git clone https://github.com/SeyyedSajjadFazeli/xai-shap-vs-lime-benchmark.git
cd xai-shap-vs-lime-benchmark
python -m venv venv
# Windows: venv\Scripts\activate | Unix: source venv/bin/activate</code></pre>

<h3>۲. نصب پکیج‌های مورد نیاز</h3>
<pre dir="ltr"><code>pip install -r requirements.txt</code></pre>

<h3>۳. اجرای پایپ‌لاین بنچمارک</h3>
<pre dir="ltr"><code>python shap_vs_lime_comparison.py</code></pre>

</div>

<hr />

<p align="center">
  Developed with ❤️ by <b>Your Name</b><br />
  <a href="https://www.linkedin.com/in/seyyed-sajjad-fazeli-a58aa1360/">🌐 LinkedIn Profile</a> │ <a href="https://github.com/SeyyedSajjadFazeli">💻 GitHub Profile</a>
</p>
