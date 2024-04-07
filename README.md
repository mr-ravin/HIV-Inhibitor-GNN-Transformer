# HIV-Inhibitor-GNN-Transformer
This repository includes code for classifying if a given molecule can act as a HIV Inhibitor, using the GNN Transformer architecture.

### Directory Structure
```
|-- dataset/
|-- result/
|-- weights/
|-- dataloader.oy
|-- model.py
|-- utils.py
|-- graphics_utils.py
|-- run.py
|-- requirements.txt
```

### Train and Test
```
python3 run.py --mode full
```

### Train only
```
python3 run.py --mode train
```

### Test only
```
python3 run.py --mode test
```

![analysis image](https://github.com/mr-ravin/HIV-Inhibitor-GNN-Transformer/blob/main/result/training_analysis.png?raw=true)

## License 
```
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
