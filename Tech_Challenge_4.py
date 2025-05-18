import cv2
import mediapipe as mp
from deepface import DeepFace
import os
import numpy as np
from tqdm import tqdm
from collections import Counter, deque

# Funcao principal para analisar um video e gerar um novo com anotacoes

def analyze_video(video_path, output_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Leitura do video de entrada
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o video.")
        return

    # Parametros do video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Contadores e variaveis auxiliares
    wave_count = 0
    handshake_count = 0
    dance_count = 0
    frame_count = 0
    anomaly_frames = []
    detected_emotions = []

    # Cooldowns para evitar contagens duplicadas
    dance_cooldown = 0
    handshake_cooldown = 0
    wave_cooldown = 0
    dance_ignore_window = 0

    # Parametros para detectar anomalias
    anomaly_emotions = ['disgust', 'fear', 'surprise']
    anomaly_threshold = 60
    pose_anomaly_threshold = 0.7

    # Historico de movimento corporal para analise de danca
    dance_motion_history = deque(maxlen=30)
    dance_movement_window = deque(maxlen=15)
    pose_anomaly_window = deque(maxlen=10)

    # Estados da frame atual
    face_detected_this_frame = False
    upper_body_visible = False
    face_bbox_height_ratio = 0.0
    hand_raised_this_frame = False

    # Funcao util para desenhar texto com fundo
    def draw_text_with_background(img, text, org, font, scale, text_color, bg_color, thickness):
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), bg_color, -1)
        cv2.putText(img, text, org, font, scale, text_color, thickness, cv2.LINE_AA)

    # Verifica se uma das maos esta levantada
    def detect_hand_raised(landmarks):
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
        return left_wrist.y < left_eye.y or right_wrist.y < right_eye.y

    # Verifica se houve movimentacao compatível com danca
    def detect_specific_dance(pose_history):
        if len(pose_history) < 10:
            return False

        left_hip_y = [pose[2][1] for pose in pose_history]
        right_hip_y = [pose[3][1] for pose in pose_history]
        left_shoulder_y = [pose[0][1] for pose in pose_history]
        right_shoulder_y = [pose[1][1] for pose in pose_history]

        hip_movement = max(left_hip_y + right_hip_y) - min(left_hip_y + right_hip_y)
        shoulder_movement = max(left_shoulder_y + right_shoulder_y) - min(left_shoulder_y + right_shoulder_y)

        return hip_movement > 0.05 and shoulder_movement > 0.05

    # Loop de leitura e analise de frames
    for _ in tqdm(range(total_frames), desc="Processando video"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        face_detected_this_frame = False
        upper_body_visible = False
        face_bbox_height_ratio = 1.0
        hand_raised_this_frame = False

        current_frame_emotions = set()
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                for face in result:
                    region = face.get('region', {})
                    if region and region.get('w', 0) > 0 and region.get('h', 0) > 0:
                        face_detected_this_frame = True
                        face_bbox_height_ratio = (region['h']) / frame.shape[0]
                        if face_bbox_height_ratio < 0.6:
                            upper_body_visible = True

                        dominant_emotion = face['dominant_emotion']
                        detected_emotions.append(dominant_emotion)
                        current_frame_emotions.add(dominant_emotion)

                        emotions = face['emotion']
                        if dance_cooldown == 0:
                            for emotion, intensity in emotions.items():
                                if emotion in anomaly_emotions and intensity >= anomaly_threshold:
                                    anomaly_frames.append(frame_count)
                                    break

                        # Mostra a emocao dominante no frame
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        draw_text_with_background(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), (36, 255, 12), 2)
        except:
            pass

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Define pontos chave do corpo
            keypoints = [
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]

            current_pose = [(lm[p].x, lm[p].y) for p in keypoints]

            # Identifica aceno
            if face_detected_this_frame and upper_body_visible and face_bbox_height_ratio < 0.6:
                if wave_cooldown == 0 and detect_hand_raised(lm):
                    wave_count += 1
                    wave_cooldown = 30
                    hand_raised_this_frame = True
                    dance_ignore_window = 20  # impede contagem de danca apos aceno

                if dance_ignore_window == 0 and dance_cooldown == 0:
                    dance_motion_history.append(current_pose)
                    if detect_specific_dance(dance_motion_history):
                        dance_count += 1
                        dance_cooldown = 30
                        dance_motion_history.clear()

            # Identifica aperto de mao (maos proximas horizontal/verticalmente)
            if handshake_cooldown == 0:
                lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
                if abs(lw.x - rw.x) < 0.05 and abs(lw.y - rw.y) < 0.05:
                    handshake_count += 1
                    handshake_cooldown = 30

        # Atualiza cooldowns
        if dance_cooldown > 0:
            dance_cooldown -= 1
        if handshake_cooldown > 0:
            handshake_cooldown -= 1
        if wave_cooldown > 0:
            wave_cooldown -= 1
        if dance_ignore_window > 0:
            dance_ignore_window -= 1

        # Mostra informacoes no video
        y_offset = 30
        draw_text_with_background(frame, f'Frames analisados: {frame_count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), (255, 255, 0), 2)
        y_offset += 30
        emotion_text = 'Expressos emocionais: ' + ', '.join(sorted(current_frame_emotions))
        draw_text_with_background(frame, emotion_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), (255, 255, 255), 2)
        y_offset += 30
        draw_text_with_background(frame, f'Anomalias: {len(anomaly_frames)}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 150), 2)
        y_offset += 30
        draw_text_with_background(frame, f'Atividade - Acenos: {wave_count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 100, 0), 2)
        y_offset += 30
        draw_text_with_background(frame, f'Atividade - Danca: {dance_count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), (200, 255, 255), 2)
        y_offset += 30
        draw_text_with_background(frame, f'Atividade - Aperto de maos: {handshake_count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (100, 100, 150), 2)

        out.write(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Gera relatorio final com estatisticas do video
    emotion_counts = Counter(detected_emotions)
    report_path = os.path.splitext(output_path)[0] + '_relatorio.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RELATORIO DE ANALISE DO VIDEO\n")
        f.write("===============================\n\n")
        f.write(f"Total de frames analisados: {frame_count}\n")
        f.write(f"Total de acenos detectados: {wave_count}\n")
        f.write(f"Total de apertos de mao detectados: {handshake_count}\n")
        f.write(f"Total de dancas detectadas: {dance_count}\n\n")
        f.write("Emocoes detectadas:\n")
        for emotion, count in emotion_counts.items():
            f.write(f"- {emotion}: {count}\n")
        f.write("\n")
        f.write(f"Total de frames com possiveis anomalias: {len(anomaly_frames)}\n")
        if anomaly_frames:
            f.write(f"Frames com anomalias: {anomaly_frames}\n")


# Execucao do script
if __name__ == '__main__':
    script_dir = os.getcwd()
    input_video_path = os.path.join(script_dir, 'video.mp4')
    output_video_path = os.path.join(script_dir, 'output_video_analysis.mp4')
    analyze_video(input_video_path, output_video_path)
