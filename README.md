
# 📊 Video Human Activity & Emotion Analyzer

Este projeto realiza a análise automática de **atividades corporais** (como aceno, dança e aperto de mãos) e **emoções faciais** em vídeos, utilizando as bibliotecas `MediaPipe`, `OpenCV` e `DeepFace`. O resultado é um novo vídeo anotado e um relatório detalhado com as estatísticas detectadas.

---

## 🧠 Tecnologias Utilizadas

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://developers.google.com/mediapipe)
- [DeepFace](https://github.com/serengil/deepface)
- [NumPy](https://numpy.org/)
- [tqdm](https://github.com/tqdm/tqdm)

---

## ⚙️ Funcionalidades

- Detecção de **acenos** (mão levantada acima dos olhos)
- Detecção de **dança** (movimentos verticais repetitivos de ombros e quadris)
- Detecção de **aperto de mãos** (mãos próximas em x e y)
- Detecção de **emoções faciais**
- Identificação de **anomalias emocionais** (surpresa, medo, nojo com intensidade alta)
- Geração de vídeo anotado
- Geração de relatório com estatísticas em `.txt`

---

## 🛠️ Como Executar

### 1. Clone o Repositório

```bash
git clone git@github.com:DanielCorreiaReis/tech-challenge-fase4.git
cd seu-repositorio
```

### 2. Instale as Dependências

Recomenda-se usar Python 3.10 ou superior.

```bash
pip install opencv-python mediapipe deepface numpy tqdm
```

> ⚠️ **Atenção:** A biblioteca `DeepFace` pode requerer bibliotecas adicionais como `tensorflow` e `keras`. Caso tenha erro, instale manualmente:
> ```bash
> pip install tensorflow keras
> ```

### 3. Coloque o vídeo a ser analisado na raiz do projeto

Nomeie o vídeo como `video.mp4` ou edite o nome dentro do código no bloco abaixo:

```python
input_video_path = os.path.join(script_dir, 'video.mp4')
```

### 4. Execute o Script

```bash
python seu_arquivo.py
```

---

## 📂 Saídas Geradas

- `output_video_analysis.mp4`: vídeo original com anotações de atividades e emoções.
- `output_video_analysis_relatorio.txt`: relatório de contagem de atividades e emoções detectadas.

---

## 📌 Observações

- O desempenho pode variar dependendo da qualidade da imagem e da posição dos participantes no vídeo.
- O DeepFace analisa emoções com base em rostos visíveis. Rostos ocultos ou fora de foco podem afetar a detecção.
- O código inclui mecanismos de *cooldown* para evitar contagens duplicadas em movimentos contínuos.
