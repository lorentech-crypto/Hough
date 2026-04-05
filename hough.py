import cv2
import numpy as np

# ###############################################
# 1. LECTURA DE LA IMAGEN                       #
# ###############################################
# Primero se carga en escala de grises
imagen_original = cv2.imread("secreto.jpg", cv2.IMREAD_GRAYSCALE)

if imagen_original is None:
    print("Error: no se pudo cargar 'secreto.jpg'. Comprueba que el fichero está en el mismo directorio.")
    exit()

cv2.imshow("1 - Imagen original", imagen_original)

# ###############################################
# 2. ECUALIZACIÓN DEL HISTOGRAMA                #
# ###############################################
# equalizeHist redistribuye los niveles de intensidad para maximizar el contraste. 
imagen_ecualizada = cv2.equalizeHist(imagen_original)

cv2.imshow("2 - Imagen ecualizada", imagen_ecualizada)

# ###############################################
# 3. DETECCIÓN DE BORDES (imagen intermedia)    #
# ###############################################
# Aplicamos el detector de Canny sobre la imagen ya ecualizada para obtener los contornos.
# Pruebas con threshold1=50, threshold2=150 y nops quedamos con 300 
bordes = cv2.Canny(imagen_ecualizada, threshold1=50, threshold2=300, apertureSize=3)

cv2.imshow("3 - Bordes (Canny)", bordes)

# ###############################################
# 4. DETECCIÓN DE LÍNEAS CON HOUGH              #
# ###############################################
# rho=1: resolución de 1 píxel en el acumulador.
# theta=pi/180: resolución de 1 grado.
# threshold=300: mínimo de votos para considerar una línea válida.

lineas = cv2.HoughLines(bordes, rho=1, theta=np.pi / 180, threshold=300)

# ###############################################
# 5. FILTRADO DE LÍNEAS SIMILARES               #
# ###############################################
# con este filtrado se descartan duplicados comparando cada candidata con las ya aceptadas.

def filtrar_lineas_similares(lineas, umbral_rho=25, umbral_theta=0.15):
    if lineas is None:
        return []

    seleccionadas = []
    for linea in lineas:
        rho, theta = linea[0]
        es_similar = False
        for r_sel, t_sel in seleccionadas:
            if abs(rho - r_sel) < umbral_rho and abs(theta - t_sel) < umbral_theta:
                es_similar = True
                break
        if not es_similar:
            seleccionadas.append((rho, theta))

    return seleccionadas


lineas_filtradas = filtrar_lineas_similares(lineas, umbral_rho=25, umbral_theta=0.15)

# ################################################
# 6. DIBUJO DE LÍNEAS SOBRE LA IMAGEN ECUALIZADA #
# ################################################
# Se pasa la ecualizada a BGR para poder dibujar en blanco (255,255,255)
imagen_con_lineas = cv2.cvtColor(imagen_ecualizada, cv2.COLOR_GRAY2BGR)

for rho, theta in lineas_filtradas:
    # Se pasan de coordenadas polares a cartesianas para cv2.line.
    # +-2000 px para que la línea atraviese toda la imagen.
    a  = np.cos(theta)
    b  = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 2000 * (-b))
    y1 = int(y0 + 2000 * (a))
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * (a))

    cv2.line(imagen_con_lineas, (x1, y1), (x2, y2), (255, 255, 255), 2)

cv2.imshow("4 - Líneas detectadas (blanco)", imagen_con_lineas)

# ###############################################
# 7. MENSAJE POR CONSOLA                        #
# ###############################################
num_lineas = len(lineas_filtradas)
print(f"Se han dibujado {num_lineas} líneas")

cv2.waitKey(0)
cv2.destroyAllWindows()
