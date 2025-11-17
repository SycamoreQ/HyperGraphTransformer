A PyTorch Geometric implementation of a novel HyperGraph Transformer (HGT) architecture for image classification, specifically applied to Magnetic Resonance Imaging (MRI) brain tumor detection. This model leverages dynamic graph construction to capture local and global relationships between image patches.

- Model Architecture & Core Concept
The HyperGraph Transformer (HGT), implemented as HyperVisionNet, is a hybrid GNN-Transformer that processes images by converting them into dynamically generated graphs.

- Key Features

Image-to-Graph Conversion: Images are divided into patches, which become the nodes of a graph.

Hybrid Connectivity: Each graph is constructed on-the-fly using two methods:

k-Nearest Neighbors (k-NN): Connects spatially or feature-similar patches.

Hyperedges: Uses clustering to define dense connections (cliques) between patches belonging to the same abstract region, significantly boosting expressivity.

GNN Core: The aggregated graph is processed by GraphSAGE (SAGEConv) layers to learn robust node representations.

Classification: Global pooling aggregates node features into a single vector for final classification into four tumor types (Giloma, Meningioma, No Tumor, Pituitary).


- Step 1 : Clone Repository 
git clone https://github.com/SycamoreQ/HyperGraphTransformer.git
cd HyperGraphTransformer


- Step 2 : Initialize virtual environment
python3 -m venv .venv
source .venv/bin/activate

- Step 3 : Install Dependencies 
pip install torch torchvision torchaudio
pip install torch-geometric
pip install wandb matplotlib pillow opencv-python

- Step 4 : Train the Model
- Set environment variable for necessary MPS fallbacks (for Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1

- Run the training script with recommended parameters
python dataloader.py \
  --data_dir "/path/to/your/dataset" \
  --wandb \
  --use_hyperedges \
  --batch_size=2 \
  --accumulation_steps=4 \
  --epochs=20

- Step 5 : Inference 
python predict.py \
  --image "/path/to/test_image.jpg" \
  --model_weights "best_hypervision.pt" \
  --k 5 \
  --num_clusters 4 \
  --device cpu
