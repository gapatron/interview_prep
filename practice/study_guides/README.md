# Study Guides

## 📄 Files

- **BFS_DFS_Study_Guide.md** / **.html** — BFS & DFS (editable / printable)
- **Binary_Search_Mastery_Guide.md** / **.html** — Binary search
- **PyTorch_Mastery_Guide.md** / **.html** — PyTorch: data, models, training, bugs (Adobe-style)
- **Advanced_Architectures_Guide.md** / **.html** — Diffusion & flow matching, Transformers, popular architectures (overview)
- **Popular_Architectures_Explained.md** — **The “why” only (no code):** residual connections, BN order, 3×3 stacks, shortcuts, VGG/ResNet/Bottleneck/Inception design and common bugs. Use with `12_popular_architectures/` challenges.
- **Computation_Exercises_Guide.md** — **Parameters, dimensions, complexity:** Conv 1D/2D/3D (output shape, params, FLOPs), Transformer and ViT shapes & params, time/space complexity, interview gotchas. Use with `week4_pytorch/14_computation_exercises/`.

## 🖨️ Converting to PDF

### Method 1: Print to PDF from Browser (Easiest!)
1. Open `BFS_DFS_Study_Guide.html` in your web browser
2. Press `Ctrl+P` (or `Cmd+P` on Mac)
3. Select "Save as PDF" as the destination
4. Print!

### Method 2: Using Pandoc (Best Quality)
```bash
# Install pandoc (if not installed)
sudo apt install pandoc texlive-xetex  # Linux
brew install pandoc                    # Mac

# Convert to PDF
cd practice/study_guides
pandoc BFS_DFS_Study_Guide.md -o BFS_DFS_Study_Guide.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V fontsize=11pt
```

### Method 3: Online Converter
1. Go to https://www.markdowntopdf.com/
2. Upload `BFS_DFS_Study_Guide.md`
3. Download the PDF

### Method 4: VS Code Extension
1. Install "Markdown PDF" extension in VS Code
2. Open `BFS_DFS_Study_Guide.md`
3. Right-click → "Markdown PDF: Export (pdf)"

## 📚 Study Guide Contents

The study guide includes:

✅ **Fill-in-the-blank exercises** - Test your understanding  
✅ **Code completion exercises** - Practice implementation  
✅ **Quizzes** - Self-assessment questions  
✅ **Practice problems** - Real interview questions  
✅ **Answer key** - Check your work  
✅ **Study checklist** - Track your progress

## 🎯 How to Use

1. **Read through** the core concepts sections
2. **Fill in** all the exercises without looking at answers
3. **Complete** the code exercises
4. **Take** the quizzes
5. **Check** your answers in the answer key
6. **Practice** implementing from memory

## 💡 Tips

- Print the study guide and work offline
- Use different colored pens for corrections
- Revisit sections you struggle with
- Time yourself on practice problems
- Implement solutions from scratch multiple times
