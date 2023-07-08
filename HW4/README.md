# Machine Learn : CNN
## Written By 練鈞揚 😎
### Py env : Anaconda Py3.11 🐍

---
### File Type
File using : ipyhb 

---

### How to use 

in this folder , you can see 4 different folder 
- `Efficientnet_v2_m`
- `ResNet18`
- `ResNet50`
- `Efficientnet_v2_l`

You can choose one of the folder to use it, or retrain the model 

**!! you need retain all model, because the train model result is so large can not upload it**

### How to load it ? 
example code

```python
# normal load model
model.load_state_dict(torch.load("model_file_name.pth"))

# if have some bug , use this 
model.load_state_dict(torch.load("model_file_name.pth"),strict=False)
```