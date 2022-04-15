# Transformer

✅ 스터디하면서 배운 것

### - Pytorch Lightning

   - Pytorch Lightning이란 pytorch 문법을 가지면서 학습코드를 pytorch보다 더 효율적으로 작성할 수 있는 파이썬 오픈소스 라이브러리
   
   - pytorch를 통해 쉽게 딥러닝 모델을 만들 수 있지만 CPU, GPU,TPU간의 변경, mixed_precision training(16bit)등의 복잡한 조건과 반복되는 코드(training, validation,test, inference)들을좀 더 효율적으로 추상화 시키자는 것을 목적으로 나오게됨

  - 즉, 이 라이브러리는 모델 코드와 엔지니어링 코드를 분리해서 코드를 깔끔하게 만들 수 있도록 해주고 16-bit training, 다중 cpu 사용 등을 포함한 다양한 학습 기법을 몇 줄의 코드로 손쉽게 사용할 수 있도록 함

  1) install & import
    
     pytorch lightning을 설치하고 import함
     
          pip install pytorch-lightning

          import pytorch_lightning as pl
          
       
  2) lightning model
  
      기존 pytorch는 DataLoader, Mode, Optimizer, Training roof 등을 전부 따로따로 코드로 구현해야했는데 pytorch lightning에서는 Lightning Model Class 안에 이 모든 것을 한번에 구현하도록 되어있음(클래스 내부에 있는 함수명은 그대로 써야하고 목적에 맞게 써야함. ex. Dataset의 init, getitem, len)
      
      torch의 nn.Module과 같이 lightning model 정의를 할 클래스에는 반드시 LightningModule을 상속 받음
          
          from efficientnet_pytorch import EfficientNet
          from pytorch_lightning.metrics.classification import AUROC
          from sklearn.metrics import roc_auc_score

          class Model(pl.LightningModule):
              def __init__(self, *args, **kwargs):
                  super().__init__()
                  self.net = EfficientNet.from_pretrained(arch, advprop=True)   # pretrained모델 생성하고 transfer learning위해 마지막 linear layer출력을 1로 바꿔줌
                  self.net._fc = nn.Linear(in_features=self.net._fc.in_features, out_features=1, bias=True)
                  
                  
       model의 입력에 대한 output을 내는 forward
      
         def forward(self, x):
              return self.net(x)
                  
       최적화를 위한 optimizer와 learning rate scheduler 초기화 및 반환    
       
          def configure_optimizers(self):
              optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
              scheduler = torch.optim.lr_scheduler.OneCycleLR(
                  max_lr=lr,
                  epochs=max_epochs,
                  optimizer=optimizer,
                  steps_per_epoch=int(len(train_dataset) / batch_size),
                  pct_start=0.1,
                  div_factor=10,
                  final_div_factor=100,
                  base_momentum=0.90,
                  max_momentum=0.95,
              )
              return [optimizer], [scheduler]       
  
       forward통해 output얻고 loss 계산하는 step 함수 
       
       여기서 parameter batch는 1 iteration에 대한 batch를 의미

          def step(self, batch):  # forward and calculate loss
              # return batch loss
              x, y  = batch
              y_hat = self(x).flatten()
              y_smo = y.float() * (1 - label_smoothing) + 0.5 * label_smoothing
              loss  = F.binary_cross_entropy_with_logits(y_hat, y_smo.type_as(y_hat),
                                                         pos_weight=torch.tensor(pos_weight))
              return loss, y, y_hat.sigmoid()   # y_hat sigmoid취해서 0-1사이 값으로 만들어줌 (나중에 accuracy계산에 사용)
  
        1 iteration에 대한 training
        
        batch 만큼 output을 얻고 loss와 accuracy return
        
           def training_step(self, batch, batch_nb):
              # hardware agnostic training
              loss, y, y_hat = self.step(batch)
              acc = (y_hat.round() == y).float().mean().item()
              tensorboard_logs = {'train_loss': loss, 'acc': acc}
              return {'loss': loss, 'acc': acc, 'log': tensorboard_logs}
  
        validation step은 1 iteration에 대한 함수 -> training step과 같은 역할
        
        validation_epoch_end는 1 epoch에 대한 함수  -> logging과 학습 과정에 대한 print를 위해 사용
        
        
           def validation_step(self, batch, batch_nb):
                loss, y, y_hat = self.step(batch)
                return {'val_loss': loss,
                        'y': y.detach(), 'y_hat': y_hat.detach()}   # detach() : 기존 tensor를 복사하는 방법 중 하나(기존 tensor에서 gradient전파가 안되는 tensor 생성)

           def validation_epoch_end(self, outputs):  # 한 에폭이 끝났을 때 실행
                avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
                y = torch.cat([x['y'] for x in outputs])
                y_hat = torch.cat([x['y_hat'] for x in outputs])
                auc = AUROC()(y_hat, y) if y.float().mean() > 0 else 0.5 # skip sanity check
                acc = (y_hat.round() == y).float().mean().item()
                print(f"Epoch {self.current_epoch} acc:{acc} auc:{auc}")
                tensorboard_logs = {'val_loss': avg_loss, 'val_auc': auc, 'val_acc': acc}
                return {'avg_val_loss': avg_loss,
                  'val_auc': auc, 'val_acc': acc,
                  'log': tensorboard_logs} 
  
        test단계를 추론 과정이기 때문에 정답이 없음  
  
           def test_step(self, batch, batch_nb):
                x, _ = batch
                y_hat = self(x).flatten().sigmoid()
                return {'y_hat': y_hat}

           def test_epoch_end(self, outputs):
                y_hat = torch.cat([x['y_hat'] for x in outputs])
                df_test['target'] = y_hat.tolist()
                N = len(glob('submission*.csv'))
                df_test.target.to_csv(f'submission{N}.csv')
                return {'tta': N}  

        각 학습 모드의 data loader를 초기화  
        
           def train_dataloader(self):
                return DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                  drop_last=True, shuffle=True, pin_memory=True)

           def val_dataloader(self):
                return DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers,
                                  drop_last=False, shuffle=False, pin_memory=True)

            def test_dataloader(self):
                return DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                  drop_last=False, shuffle=False, pin_memory=False)  

        지금까지가 LightningModule내에서 정의되는 메서드

        이렇게 재정의를 하고나면 바로 학습 가능
          
        pytorch lightning에서의 학습 코드는 다음과 같음
          
            checkpoint_callback = pl.callbacks.ModelCheckpoint('{epoch:02d}_{val_auc:.4f}',
                                                              save_top_k=1, monitor='val_auc', mode='max')  # val_auc가 최대값이 되면 저장되도록함                                                 
            trainer = pl.Trainer(
                tpu_cores=tpu_cores,
                gpus=gpus,
                precision=16 if gpus else 32,
                max_epochs=max_epochs,
                num_sanity_val_steps=1 if debug else 0,  # catches any bugs in your validation without having to wait for the first validation check. 
                checkpoint_callback=checkpoint_callback
                )

            trainer.fit(model)    # 학습 시작      
  
  ➡️ **pytorch lightning의 장점은 세부적인 high-level코드를 작성할 때 좀 더 정돈되고 간결하게 작성 가능**
  
  
   **[참고]**

   https://visionhong.tistory.com/30
