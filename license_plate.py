import cv2
import numpy as np
import char_recognition as cr

cr.initializeModel('siec_v3.h5')

# STAŁE
lp_width = 520
lp_height = 114
lp_char_height = 80

lp_width_max = lp_width * 3 / 2
lp_width_min = lp_width * 1 / 2
lp_height_max = lp_height * 3 / 2
lp_height_min = lp_height * 1 / 2

# Malinowska
lp_m = (2 * lp_width + 2 * lp_height) / (2 * np.sqrt(np.pi * (lp_width * lp_height))) - 1
lp_m_max = (2 * lp_width_max + 2 * lp_height_min) / (2 * np.sqrt(np.pi * (lp_width_max * lp_height_min))) - 1
lp_m_min = (2 * lp_width_min + 2 * lp_height_max) / (2 * np.sqrt(np.pi * (lp_width_min * lp_height_max))) - 1

# Stosunek Boków
lp_wh = lp_width / lp_height
lp_wh_min = lp_width_min / lp_height_max
lp_wh_max = lp_width_max / lp_height_min

# Stosunek wysokości znaku do wysokości tablicy
lp_char_ratio_max = (lp_char_height / lp_height) * 1.5
lp_char_ratio_min = (lp_char_height / lp_height) * 0.8


# Metoda zwracająca kontury wewnątrz danego konturu
def getKidContours(contours, hierarchy, index, all=False):
    result = []
    indexes = []
    pom = hierarchy[0][index][2]
    if (pom == -1):
        return result, indexes

    # next
    while pom != -1:
        indexes.append(pom)
        result.append(contours[pom])
        pom = hierarchy[0][pom][0]

    # previous
    pom = hierarchy[0][hierarchy[0][index][2]][1]
    while pom != -1:
        indexes.append(pom)
        result.append(contours[pom])
        pom = hierarchy[0][pom][1]

    # Kids
    if (all):
        for i in indexes:
            kids_contours, kids_indexes = getKidContours(contours, hierarchy, i)
            result.extend(kids_contours)
            indexes.extend(kids_indexes)

    return result, indexes


# Wyświetlanie
def show(img, name):
    cv2.imshow(name, cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    cv2.waitKey()
    cv2.destroyAllWindows()


# Preprocesing wczytanego zdjęcia gray->blur->Canny
def preprocessing(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_canny = cv2.Canny(img_blur, 87, 200)
    return img_canny


# Preprocesing analizowanego znaku przed podaniem na sieć
def charPreprocessing(img):
    char_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, char_threshold = cv2.threshold(char_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    char_threshold_rgb = cv2.cvtColor(char_threshold, cv2.COLOR_GRAY2RGB)
    return char_threshold_rgb


# Sprawdzenie czy dany kontur przypomina prostokąt tablicy rejestacyjnej
def isPlate(con):
    S = cv2.contourArea(con)
    L = cv2.arcLength(con, True)
    rect = cv2.minAreaRect(con)
    w_rect = rect[1][0]
    h_rect = rect[1][1]
    S_pred = w_rect * h_rect
    L_pred = 2 * w_rect + 2 * h_rect

    # Nie zerowy obwód oraz pole większe od 100
    if (S <= 100 or L == 0):
        return False

    # Czy pole konturu jest mniej więcej takie same jak pole prostokątą opisującego kontur
    if (not (S > S_pred * 0.7 and S < S_pred * 1.3)):
        return False

    # Czy obwód konturu jest mniej więcej taki sam jak obwód prostokąta opisującego kontur
    if (not (L > L_pred * 0.7 and L < L_pred * 1.3)):
        return False

    # Sprawdzenie czy stosunek boków prostokąta jest mniej więcej zgodny ze stosunkiem rzeczywistej tablicy rejestracyjnej
    if (not ((max(w_rect, h_rect) / min(h_rect, w_rect) < lp_wh_max) and (
            max(w_rect, h_rect) / min(h_rect, w_rect) > lp_wh_min))):
        return False

    # wsp Malinowskiej dla tablicy
    M = (L / (2 * np.sqrt(np.pi * S))) - 1
    if (not (M < max(lp_m_max, lp_m, lp_wh_min) and M > min(lp_m_max, lp_m, lp_wh_min))):
        return False

    return True


# Sprawdzanie czy kontur jest znakiem
def isChar(plateCon, charCon):
    S = cv2.contourArea(plateCon)

    rect = cv2.minAreaRect(plateCon)
    w_rect = rect[1][0]
    h_rect = rect[1][1]

    c_rect = cv2.minAreaRect(charCon)
    h_c_rect = c_rect[1][1]
    w_c_rect = c_rect[1][0]

    # Czy pole znaku ma odpowiednią wartość w porównaniu do pola tablicy rejestracyjnej
    s_kc = w_c_rect * h_c_rect
    if (not (s_kc > S * 0.01 and s_kc < S * 0.15)):
        return False

    # Czy wysokość znaku jest odpowiedania do wysokości
    h_ratio = max(h_c_rect, w_c_rect) / min(h_rect, w_rect)
    if (not (h_ratio < lp_char_ratio_max and h_ratio > lp_char_ratio_min)):
        return False

    return True


# Wyszukuje napisu w konturze
def getPlateLabel(img, allContours, hierarchy, usedContours, plateIndex, index=-2):
    if index == -2: index = plateIndex
    done = False
    plateLabel = ''
    potential_char_contours = []
    potential_char_indexes = []

    if index in usedContours: return [], [], [], False

    if index == -1: return [], [], [], False

    kid_contours, kid_indexes = getKidContours(allContours, hierarchy, index)

    for i in range(len(kid_indexes)):
        if isChar(allContours[plateIndex], allContours[kid_indexes[i]]):
            potential_char_contours.append(allContours[kid_indexes[i]])
            potential_char_indexes.append(kid_indexes[i])

    if len(potential_char_indexes) > 3:
        sorted_chars = sorted(potential_char_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        for char in sorted_chars:
            x, y, w, h = cv2.boundingRect(char)
            char_image = charPreprocessing(img[y - 5:y + h + 5, x - 5:x + w + 5])
            char, pred = cr.recognize(char_image)
            plateLabel = plateLabel + char

        done = True
    else:
        for i in range(len(kid_indexes)):
            plateLabel, potential_char_contours, potential_char_indexes, done = \
                getPlateLabel(img, allContours, hierarchy, usedContours, plateIndex, kid_indexes[i])
            if done: break

    return plateLabel, potential_char_contours, potential_char_indexes, done


# Główna metoda odpowiedzialna za rozpoznawanie tablic rejestracyjnych na zdjęciach
def recognize_license_plate(readpath=None, img=None, savepath=None, drawRectWithDesc=False, drawContours=False,
                            drawChars=False, show_img=False):
    img_entry = None
    if (readpath != None): img_entry = cv2.imread(readpath)
    if (img != None): img_entry = img
    if (img is None and readpath is None): raise Exception("Failed to load image")

    img_preprocessed = preprocessing(img_entry)

    all_contours, hierarchy = cv2.findContours(img_preprocessed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_result = img_entry.copy()

    usedContours = []
    labels = []

    for index in range(len(all_contours)):
        if index in usedContours: continue

        con = all_contours[index]

        if not isPlate(con): continue

        plateLabel, char_contours, char_indexes, done = getPlateLabel(img_entry,
                                                                      all_contours, hierarchy, usedContours, index)
        if done:
            _, contoursIndexes = getKidContours(all_contours, hierarchy, index, all=True)
            usedContours.extend(contoursIndexes)
            usedContours.append(index)
            labels.append(plateLabel)
        else:
            continue

        x, y, w, h = cv2.boundingRect(con)

        if drawRectWithDesc:
            cv2.putText(img_result, plateLabel, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (36, 255, 12), 3)
            cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if drawContours: cv2.drawContours(img_result, [con], -1, (0, 255, 0), 2)
        if drawChars: cv2.drawContours(img_result, char_contours, -1, (0, 0, 255), 2)

    if (savepath != None): cv2.imwrite(savepath, img_result)
    if show_img: show(img_result, "LicensePlateRecognition")

    return img_result, labels


if __name__ == "__main__":
    tryb = 1
    if (tryb == 1):
        recognize_license_plate(readpath="in/17.jpg", show_img=True, drawRectWithDesc=True)
    else:
        for i in range(23):
            recognize_license_plate(readpath="in/" + str(i + 1) + ".jpg",
                                    savepath="out/" + str(i + 1) + ".jpg", drawRectWithDesc=True)

    cv2.waitKey()
    cv2.destroyAllWindows()
