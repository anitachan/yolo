
from recognition import recolectar_imagenes, generar_base_de_datos, reconocimiento_tiempo_real
from utils import detectar_resoluciones_soportadas, RESOLUCIONES

RESOLUCION_ACTUAL = "2"

def menu():
    global RESOLUCION_ACTUAL
    while True:
        current_res = RESOLUCIONES.get(RESOLUCION_ACTUAL, (1280, 720))
        print("\n=== RECONOCIMIENTO FACIAL ===")
        print("Resolución actual: {}x{}".format(*current_res))
        print("1. Recolectar imágenes de usuario")
        print("2. Generar base de datos")
        print("3. Reconocimiento en tiempo real")
        print("4. Cambiar resolución")
        print("5. Detectar resoluciones soportadas")
        print("6. Salir")
        opcion = input("Opción: ").strip()

        if opcion == "1":
            nombre = input("Nombre del usuario: ").strip()
            recolectar_imagenes(nombre)
        elif opcion == "2":
            generar_base_de_datos()
        elif opcion == "3":
            reconocimiento_tiempo_real()
        elif opcion == "4":
            print("\nSelecciona resolución:")
            for key, (w, h) in RESOLUCIONES.items():
                print(f"{key}. {w}x{h}")
            nueva = input("Opción: ").strip()
            if nueva in RESOLUCIONES:
                RESOLUCION_ACTUAL = nueva
                print("[✓] Resolución actualizada")
            else:
                print("[X] Opción inválida.")
        elif opcion == "5":
            resoluciones = detectar_resoluciones_soportadas()
            print("[INFO] Resoluciones disponibles:")
            for w, h in resoluciones:
                print(f" - {w}x{h}")
        elif opcion == "6":
            print("Saliendo...")
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    menu()
