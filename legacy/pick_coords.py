import pygame
import sys

# Datei deiner Vorlage
IMAGE_FILE = "risk_map.png"  # benenne dein Bild so oder passe den Namen an

WINDOW_WIDTH, WINDOW_HEIGHT = 1600, 900

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("RISIKO-Koordinaten-Picker")

    font = pygame.font.SysFont("arial", 20)

    # Bild laden
    try:
        img = pygame.image.load(IMAGE_FILE).convert()
    except pygame.error as e:
        print(f"Fehler beim Laden von {IMAGE_FILE}: {e}")
        sys.exit(1)

    # Bild proportional auf Fenster skalieren
    img_rect = img.get_rect()
    scale_x = WINDOW_WIDTH / img_rect.width
    scale_y = WINDOW_HEIGHT / img_rect.height
    scale = min(scale_x, scale_y)
    new_size = (int(img_rect.width * scale), int(img_rect.height * scale))
    img = pygame.transform.smoothscale(img, new_size)
    img_rect = img.get_rect()
    img_rect.center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)

    print("Links-Klick auf einen Kreis setzt einen Punkt.")
    print("Konsole zeigt: (x, y) Koordinaten relativ zum 1600x900-Fenster.")
    print("ESC oder Fenster schließen zum Beenden.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Linksklick
                mx, my = event.pos
                print(f"Clicked at: ({mx}, {my})")

        screen.fill((0, 0, 0))
        screen.blit(img, img_rect)

        # kleine Hilfe-Anzeige
        text = font.render("Links-Klick: Koordinate in Konsole | ESC: Exit", True, (255, 255, 255))
        screen.blit(text, (10, 10))

        pygame.display.flip()

    pygame.quit()


if __name__ == \"__main__\":
    main()
