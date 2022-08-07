import cv2
import mediapipe as mp
import math
import time

# video captura
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # ancho ventana
cap.set(6, 720) # alto ventana

# variables de conteo
parpadeo = False
conteo = 0
tiempo = 0
inicio = 0
final = 0
conteo_sue = 0
muestra = 0

# funcion de dibujo
mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness = 1, circle_radius = 1) # configuración del dibujo

# objeto donde se almacena la malla facial
mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces = 4)

# while principal
while True:
    ret, frame = cap.read()
    #correción de color
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #observamos los resultados
    resultados = MallaFacial.process(frameRGB)

    # listas para almacenar resultados
    px = []
    py = []
    lista = []
    r = 5
    t = 3

    if resultados.multi_face_landmarks: # existe un rostro
        for rostros in resultados.multi_face_landmarks:
            mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACEMESH_CONTOURS, ConfDibu, ConfDibu)

            # extraer los puntos del rostro detectado
            for id,puntos in enumerate(rostros.landmark):
                al, an, c = frame.shape
                x, y = int(puntos.x * an), int(puntos.y * al)
                px.append(x)
                py.append(y)
                lista.append([id, x, y])
                if len(lista) == 468:
                    # ojo derecho
                    x1, y1 = lista[145][1:]
                    x2, y2 = lista[159][1:]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    # cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), t)
                    # cv2.circle(frame, (x1, y1), r, (0, 0, 0), cv2.FILLED)
                    # cv2.circle(frame, (x2, y2), r, (0, 0, 0), cv2.FILLED)
                    # cv2.circle(frame, (cx, cy), r, (0, 0, 0), cv2.FILLED)
                    longitud1 = math.hypot(x2 - x1, y2 -y1)
                    # ojo izquierdo
                    x3, y3 = lista[374][1:]
                    x4, y4 = lista[386][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                    longitud2 = math.hypot(x4 - x3, y4 -y3)
                    # conteo de parpadeos
                    cv2.putText(frame, f'Parpadeos: {int(conteo)}', (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(frame, f'Micro sueno: {int(conteo_sue)}', (780, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.putText(frame, f'Duracion: {int(muestra)}', (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    if longitud1 <= 14 and longitud2 <= 14 and parpadeo == False: # parpadeo
                        conteo = conteo + 1
                        parpadeo = True
                        inicio = time.time()
                    elif longitud1 > 14 and longitud2 > 14 and parpadeo == True: # seguridad de parpadeo
                        parpadeo = False
                        final = time.time()
                    # temporizador
                    tiempo = round(final - inicio, 0)
                    # contador micro sueño
                    if tiempo >= 3:
                        conteo_sue = conteo_sue + 1
                        muestra = tiempo
                        inicio = 0
                        final = 0
    # mostramos el frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # si pulsa q se rompe el ciclo
    if key == ord("q"):
        break