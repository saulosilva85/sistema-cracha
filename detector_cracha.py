import cv2
import time
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# CONFIGURAÇÕES DO SISTEMA
# ==============================================================================
VIDEO_SOURCE = 0  # 0 para Webcam local, ou URL RTSP da câmera IP
COOLDOWN_NOTIFICACAO = 10
ULTIMO_ALERTA = 0

def enviar_notificacao(mensagem):
    print(f"\n[🚨 ALERTA DE SEGURANÇA] {mensagem}\n")

def verificar_presenca_cracha(crop_peito):
    """
    Analisa a região do peito procurando estritamente por um CARTÃO DE CRACHÁ
    (superfície retangular fechada) e descarta cordões, fitas ou sombras.
    """
    if crop_peito.size == 0 or crop_peito.shape[0] < 40 or crop_peito.shape[1] < 40:
        return False, None

    altura_peito, largura_peito = crop_peito.shape[:2]
    area_peito = altura_peito * largura_peito

    # --- 1. MÁSCARA DE PELE (Para evitar falsos positivos no corpo sem camisa) ---
    hsv = cv2.cvtColor(crop_peito, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)

    # --- 2. PROCESSAMENTO DE BORDAS E CONTORNOS ---
    gray = cv2.cvtColor(crop_peito, cv2.COLOR_BGR2GRAY)
    
    # Suavização para eliminar ruídos finos (como a textura do cordão)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny com limiares equilibrados
    edges = cv2.Canny(blurred, 50, 150)

    # Fechar pequenos buracos para formar retângulos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Encontrar contornos na imagem
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Um cartão de crachá real deve ter tamanho relevante no peito (entre 2.5% e 30% da ROI)
        if (area_peito * 0.025) < area < (area_peito * 0.30):
            x, y, w, h = cv2.boundingRect(cnt)
            
            # --- REGRA ANTI-CORDÃO: Ignora objetos muito finos/compridos ---
            # O cartão do crachá precisa ter uma largura e altura mínimas
            if w < 25 or h < 30:
                continue

            # Verificação da proporção do retângulo (Vertical ou Horizontal)
            proporcao = float(h) / w if w > 0 else 0
            proporcao_inv = float(w) / h if h > 0 else 0

            # Proporções típicas de cartões ID (ex: 8.5 x 5.5 cm dá ~1.35 a 1.8)
            if (1.15 <= proporcao <= 2.2) or (1.15 <= proporcao_inv <= 2.2):
                
                # --- VERIFICAÇÃO DE CONTEÚDO (Não pode ser apenas pele) ---
                roi_skin = mask_skin[y:y+h, x:x+w]
                percentual_pele = (cv2.countNonZero(roi_skin) / (w * h)) * 100

                # Se a região delimitada for composta por mais de 50% de pele, ignora
                if percentual_pele > 50.0:
                    continue

                # --- VERIFICAÇÃO DE DETALHES INTERNOS (Foto / Texto no cartão) ---
                roi_interna = gray[y:y+h, x:x+w]
                if roi_interna.size > 0 and np.std(roi_interna) > 20:
                    return True, (x, y, w, h)

    return False, None

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    global ULTIMO_ALERTA
    
    print("[INFO] Carregando modelo YOLOv8...")
    model = YOLO("yolov8n.pt")
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERRO] Não foi possível acessar a câmera.")
        return

    print("[INFO] Câmera iniciada com sucesso. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Filtrar apenas classe "person" (0) com confiança superior a 55%
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.55:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # ==========================================================
                    # ROI CENTRALIZADA DO PEITO:
                    # - Começa abaixo do pescoço (55% a 92% da altura)
                    # - Corta 15% das laterais para focar no centro do tórax
                    # ==========================================================
                    altura_pessoa = y2 - y1
                    largura_pessoa = x2 - x1

                    peito_y1 = max(0, y1 + int(altura_pessoa * 0.55))
                    peito_y2 = min(frame.shape[0], y1 + int(altura_pessoa * 0.92))
                    
                    margem_x = int(largura_pessoa * 0.15)
                    peito_x1 = max(0, x1 + margem_x)
                    peito_x2 = min(frame.shape[1], x2 - margem_x)

                    crop_peito = frame[peito_y1:peito_y2, peito_x1:peito_x2]

                    tem_cracha, bbox_cracha = verificar_presenca_cracha(crop_peito)

                    if tem_cracha:
                        cor = (0, 255, 0)  # Verde
                        status = "CRACHA DETECTADO!"
                        
                        # Desenha a caixa verde exatamente no cartão do crachá
                        if bbox_cracha is not None:
                            cx, cy, cw, ch = bbox_cracha
                            cracha_x1 = peito_x1 + cx
                            cracha_y1 = peito_y1 + cy
                            cracha_x2 = cracha_x1 + cw
                            cracha_y2 = cracha_y1 + ch
                            
                            cv2.rectangle(frame, (cracha_x1, cracha_y1), (cracha_x2, cracha_y2), (0, 255, 0), 2)
                            cv2.putText(frame, "CARTAO", (cracha_x1, cracha_y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cor = (0, 0, 255)  # Vermelho
                        status = "CRACHA NAO DETECTADO!"

                        tempo_atual = time.time()
                        if tempo_atual - ULTIMO_ALERTA > COOLDOWN_NOTIFICACAO:
                            enviar_notificacao(f"Crachá não detectado no funcionário às {time.strftime('%H:%M:%S')}!")
                            ULTIMO_ALERTA = tempo_atual

                    # Desenhar caixas de monitoramento na pessoa
                    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                    cv2.rectangle(frame, (peito_x1, peito_y1), (peito_x2, peito_y2), (255, 255, 0), 1)  # ROI Ciano
                    
                    # Fundo preto atrás do texto
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + 310, y1), (0, 0, 0), -1)
                    cv2.putText(frame, f"{status}", (x1 + 5, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

        cv2.imshow("Monitoramento de Cracha - Câmera IP / Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()