import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import random


image_path = "C:/Users/Al Hikmah Computer/Desktop/PYTHON/pyautogui/az.jpg"
def add_paper_texture(img, intensity=10):
        """Simulate paper/canvas texture by adding noise."""
        noise = np.random.normal(loc=0, scale=intensity, size=img.shape).astype(np.float32)
        textured = img.astype(np.float32) + noise
        return np.clip(textured, 0, 255).astype(np.uint8)

def directional_blur(img, kernel_size=15, angle=0):
        """Apply directional blur simulating brush stroke direction."""
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        angle_rad = np.deg2rad(angle)
        sin_a, cos_a = np.sin(angle_rad), np.cos(angle_rad)
        for i in range(kernel_size):
            x = center + int((i - center) * cos_a)
            y = center + int((i - center) * sin_a)
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        kernel /= kernel.sum()
        return cv2.filter2D(img, -1, kernel)


def blend_edges(img, gray, edge_strength=0.2):
    edges = cv2.Canny(gray, 40, 90)
    edges = cv2.dilate(edges, np.ones((1, 1), np.uint8))
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.subtract(img, (edges_rgb * edge_strength).astype(np.uint8))


class EnhancedArtisticConverter:
    """Enhanced artistic effects converter with 260+ different styles including anime, paint, and Ghibli styles"""
    
    def __init__(self):
        self.effects = {
            # orginal effects (1-35)
            1: ("Pencil Sketch", self.pencil_sketch),
            2: ("Charcoal Drawing", self.charcoal_drawing),
            3: ("Ink Sketch", self.ink_sketch),
            4: ("Oil Painting", self.oil_painting),
            5: ("Watercolor", self.watercolor),
            6: ("Acrylic Paint", self.acrylic_paint),
            7: ("Pastel Drawing", self.pastel_drawing),
            8: ("Cross Hatching", self.cross_hatching),
            9: ("Stippling", self.stippling),
            10: ("Conte Crayon", self.conte_crayon),
            11: ("Colored Pencil", self.colored_pencil),
            12: ("Marker Drawing", self.marker_drawing),
            13: ("Chalk Drawing", self.chalk_drawing),
            14: ("Ballpoint Pen", self.ballpoint_pen),
            15: ("Scratchboard", self.scratchboard),
            16: ("Lithograph", self.lithograph),
            17: ("Woodcut Print", self.woodcut_print),
            18: ("Linocut", self.linocut),
            19: ("Etching", self.etching),
            20: ("Night Vision", self.night_vision),
            21: ("Thermal Vision", self.thermal_vision),
            22: ("X-Ray Effect", self.xray_effect),
            23: ("Neon Glow", self.neon_glow),
            24: ("Cyberpunk", self.cyberpunk),
            25: ("Vintage Sepia", self.vintage_sepia),
            26: ("Blueprint", self.blueprint),
            27: ("Pop Art", self.pop_art),
            28: ("Comic Book", self.comic_book),
            29: ("Anime Style", self.anime_style),
            30: ("Abstract Art", self.abstract_art),
            31: ("Impressionist", self.impressionist),
            32: ("Pointillism", self.pointillism),
            33: ("Expressionist", self.expressionist),
            34: ("Cubist", self.cubist),
            35: ("Art Nouveau", self.art_nouveau),
            
            # anime type (36-50)
            36: ("Shoujo Anime", self.shoujo_anime),
            37: ("Shounen Anime", self.shounen_anime),
            38: ("Kawaii Style", self.kawaii_style),
            39: ("Chibi Style", self.chibi_style),
            40: ("Cel Shading", self.cel_shading),
            41: ("Manga Black & White", self.manga_bw),
            42: ("Anime Screenshot", self.anime_screenshot),
            43: ("Visual Novel Style", self.visual_novel),
            44: ("Moe Style", self.moe_style),
            45: ("Tsundere Aesthetic", self.tsundere_aesthetic),
            46: ("Dark Anime", self.dark_anime),
            47: ("Magical Girl", self.magical_girl),
            48: ("Mecha Anime", self.mecha_anime),
            49: ("Slice of Life", self.slice_of_life),
            50: ("Anime Portrait", self.anime_portrait),
            
            # paint style (51-65)
            51: ("Fresco Painting", self.fresco_painting),
            52: ("Tempera Paint", self.tempera_paint),
            53: ("Gouache Style", self.gouache_style),
            54: ("Encaustic Wax", self.encaustic_wax),
            55: ("Chinese Ink Wash", self.chinese_ink_wash),
            56: ("Japanese Sumi-e", self.japanese_sumi_e),
            57: ("Dutch Masters", self.dutch_masters),
            58: ("Fauvism", self.fauvism),
            59: ("Neo-Impressionism", self.neo_impressionism),
            60: ("Abstract Expressionism", self.abstract_expressionism),
            61: ("Color Field Painting", self.color_field_painting),
            62: ("Plein Air", self.plein_air),
            63: ("Baroque Painting", self.baroque_painting),
            64: ("Renaissance Style", self.renaissance_style),
            65: ("Modern Abstract", self.modern_abstract),
            
            # studio style (66-80)
            66: ("Ghibli Landscape", self.ghibli_landscape),
            67: ("Ghibli Character", self.ghibli_character),
            68: ("Spirited Away Style", self.spirited_away_style),
            69: ("My Neighbor Totoro", self.totoro_style),
            70: ("Princess Mononoke", self.mononoke_style),
            71: ("Howl's Moving Castle", self.howls_castle_style),
            72: ("Castle in the Sky", self.castle_sky_style),
            73: ("Kiki's Delivery Service", self.kikis_style),
            74: ("Ponyo Style", self.ponyo_style),
            75: ("The Wind Rises", self.wind_rises_style),
            76: ("Ghibli Sky", self.ghibli_sky),
            77: ("Ghibli Forest", self.ghibli_forest),
            78: ("Ghibli Magic", self.ghibli_magic),
            79: ("Ghibli Nostalgia", self.ghibli_nostalgia),
            80: ("Ghibli Dreams", self.ghibli_dreams),
            81: ("Thick Impasto", self.thick_impasto),
            82: ("Palette Knife", self.palette_knife),
            83: ("Wet on Wet", self.wet_on_wet),
            84: ("Dry Brush", self.dry_brush),
            85: ("Glazing Layers", self.glazing_layers),
            86: ("Acrylic Pour", self.acrylic_pour),
            87: ("Scumbling", self.scumbling),
            88: ("Color Blocking", self.color_blocking),
            89: ("Textured Canvas", self.textured_canvas),
            90: ("Heavy Body Acrylic", self.heavy_body_acrylic),
            91: ("Fluid Acrylic", self.fluid_acrylic),
            92: ("Stippled Acrylic", self.stippled_acrylic),
            93: ("Crosshatched Acrylic", self.crosshatched_acrylic),
            94: ("Acrylic Wash", self.acrylic_wash),
            95: ("Color Mixing", self.color_mixing),
            96: ("Bold Strokes", self.bold_strokes),
            97: ("Layered Acrylic", self.layered_acrylic),
            98: ("Gestural Painting", self.gestural_painting),
            99: ("Acrylic Medium Gel", self.acrylic_medium_gel),
            100: ("Fan Brush Texture", self.fan_brush_texture),
            101: ("Acrylic Sgraffito", self.acrylic_sgraffito),
            102: ("Thick Paint Buildup", self.thick_paint_buildup),
            103: ("Loose Brushwork", self.loose_brushwork),
            104: ("Acrylic Underpainting", self.acrylic_underpainting),
            105: ("Broken Color", self.broken_color),
            106: ("Acrylic Glazing", self.acrylic_glazing),
            107: ("Painterly Edges", self.painterly_edges),
            108: ("Acrylic Pouring Cells", self.acrylic_pouring_cells),
            109: ("High Key Acrylic", self.high_key_acrylic),
            110: ("Low Key Acrylic", self.low_key_acrylic),
            111: ("Expressive Color", self.expressive_color),
            112: ("Acrylic Collage", self.acrylic_collage),
            113: ("Rhythmic Brushwork", self.rhythmic_brushwork),
            114: ("Acrylic Transparency", self.acrylic_transparency),
            115: ("Directional Strokes", self.directional_strokes),
            116: ("Acrylic Spray", self.acrylic_spray),
            117: ("Color Temperature Mix", self.color_temperature_mix),
            118: ("Acrylic Sketch Underlay", self.acrylic_sketch_underlay),
            119: ("Loose Color Application", self.loose_color_application),
            120: ("Acrylic Texture Paste", self.acrylic_texture_paste),
            121: ("Complementary Color Play", self.complementary_color_play),
            122: ("Energetic Brushwork", self.energetic_brushwork),
            123: ("Acrylic Color Studies", self.acrylic_color_studies),
            124: ("Monochromatic Acrylic", self.monochromatic_acrylic),
            125: ("Acrylic Washes Layered", self.acrylic_washes_layered),
            126: ("Spontaneous Application", self.spontaneous_application),
            127: ("Acrylic Resist Technique", self.acrylic_resist_technique),
            128: ("Bold Color Contrasts", self.bold_color_contrasts),
            129: ("Acrylic Dry Layering", self.acrylic_dry_layering),
            130: ("Contemporary Acrylic", self.contemporary_acrylic),
            131: ("Gaussian Blur", self.gaussian_blur_effect),
            132: ("Motion Blur", self.motion_blur_effect),
            133: ("Emboss Filter", self.emboss_filter),
            134: ("Edge Enhance", self.edge_enhance),
            135: ("Find Edges", self.find_edges_filter),
            136: ("Radial Blur", self.radial_blur),
            137: ("Lens Flare", self.lens_flare_effect),
            138: ("Plastic Wrap", self.plastic_wrap_effect),
            139: ("Chrome Effect", self.chrome_effect),
            140: ("Glowing Edges", self.glowing_edges),
# illustartor styles
            141: ("Vector Illustration", self.vector_illustration),
            142: ("Flat Design", self.flat_design_style),
            143: ("Line Art", self.line_art_illustration),
            144: ("Watercolor Illustration", self.watercolor_illustration),
            145: ("Pen & Ink", self.pen_ink_illustration),
            146: ("Digital Painting", self.digital_painting_style),
            147: ("Cartoon Style", self.cartoon_illustration),
            148: ("Sketch Illustration", self.sketch_illustration),
            149: ("Minimalist Style", self.minimalist_illustration),
            150: ("Isometric Style", self.isometric_illustration),
            151: ("Synthwave", self.synthwave_style),
            152: ("Vaporwave", self.vaporwave_style),
            153: ("Glitch Art", self.glitch_art_style),
            154: ("Holographic", self.holographic_effect),
            155: ("Matrix Style", self.matrix_digital),
            156: ("Tron Legacy", self.tron_legacy_style),
            157: ("Blade Runner", self.blade_runner_aesthetic),
            158: ("Ghost in Shell", self.ghost_shell_style),
            159: ("Akira Cyberpunk", self.akira_cyberpunk),
            160: ("Neon Noir", self.neon_noir_style),

            # vintage  (161-175)
            161: ("Film Noir", self.film_noir_style),
            162: ("Art Deco", self.art_deco_style),
            163: ("Vintage Poster", self.vintage_poster),
            164: ("Pin-up Style", self.pinup_style),
            165: ("50s Americana", self.americana_50s),
            166: ("60s Psychedelic", self.psychedelic_60s),
            167: ("70s Disco", self.disco_70s_style),
            168: ("80s Neon", self.neon_80s_style),
            169: ("90s Grunge", self.grunge_90s_style),
            170: ("Polaroid Vintage", self.polaroid_vintage),
            171: ("Retro Futurism", self.retro_futurism),
            172: ("Mid-Century Modern", self.mid_century_modern),
            173: ("Victorian Gothic", self.victorian_gothic),
            174: ("Steampunk", self.steampunk_style),
            175: ("Dieselpunk", self.dieselpunk_style),

            # old styles (176-185)
            176: ("Ancient Manuscript", self.ancient_manuscript),
            177: ("Medieval Illumination", self.medieval_illumination),
            178: ("Tapestry Weave", self.tapestry_weave),
            179: ("Fresco Ancient", self.ancient_fresco),
            180: ("Byzantine Icon", self.byzantine_icon),
            181: ("Egyptian Hieroglyphic", self.egyptian_style),
            182: ("Greek Pottery", self.greek_pottery_style),
            183: ("Roman Mosaic", self.roman_mosaic),
            184: ("Cave Painting", self.cave_painting_style),
            185: ("Parchment Scroll", self.parchment_scroll),

# mordern styles (186-200)
            186: ("Low Poly", self.low_poly_style),
            187: ("Pixel Art", self.pixel_art_style),
            188: ("Voxel Art", self.voxel_art_style),
            189: ("Wireframe", self.wireframe_style),
            190: ("Procedural", self.procedural_style),
            191: ("Photogrammetry", self.photogrammetry_style),
            192: ("Neural Style", self.neural_style_transfer),
            193: ("Deep Dream", self.deep_dream_style),
            194: ("Algorithmic Art", self.algorithmic_art),
            195: ("Generative Art", self.generative_art_style),
            196: ("Glitch Datamosh", self.glitch_datamosh),
            197: ("Chromatic Aberration", self.chromatic_aberration),
            198: ("Scan Lines", self.scan_lines_effect),
            199: ("CRT Monitor", self.crt_monitor_effect),
            200: ("Hologram Interference", self.hologram_interference),


            201: ("Arabic Calligraphy", self.arabic_calligraphy),
            202: ("Islamic Geometric", self.islamic_geometric),
            203: ("Arabesque Pattern", self.arabesque_pattern),
            204: ("Mosque Architecture", self.mosque_architecture),
            205: ("Persian Miniature", self.persian_miniature),
            206: ("Ottoman Art", self.ottoman_art),
            207: ("Mamluk Style", self.mamluk_style),
            208: ("Moorish Design", self.moorish_design),
            209: ("Kufic Script", self.kufic_script),
            210: ("Nastaliq Calligraphy", self.nastaliq_calligraphy),
            211: ("Mihrab Style", self.mihrab_style),
            212: ("Islamic Tile Work", self.islamic_tilework),
            213: ("Crescent Moon Art", self.crescent_moon_art),
            214: ("Star Pattern", self.star_pattern),
            215: ("Islamic Border", self.islamic_border),
            216: ("Quranic Manuscript", self.quranic_manuscript),
            217: ("Madrasa Style", self.madrasa_style),
            218: ("Islamic Garden", self.islamic_garden),
            219: ("Sufi Mystical", self.sufi_mystical),
            220: ("Hajj Pilgrimage", self.hajj_pilgrimage),

            221: ("Oil Paint Filter", self.oil_paint_filter),
222: ("Dry Brush Filter", self.dry_brush_filter),
223: ("Palette Knife Filter", self.palette_knife_filter),
224: ("Watercolor Filter", self.watercolor_filter),
225: ("Sponge Filter", self.sponge_filter),
226: ("Poster Edges", self.poster_edges),
227: ("Cutout Filter", self.cutout_filter),
228: ("Torn Edges", self.torn_edges),
229: ("Rough Pastels", self.rough_pastels),
230: ("Smudge Stick", self.smudge_stick),
231: ("Angled Strokes", self.angled_strokes),
232: ("Crosshatch Filter", self.crosshatch_filter),
233: ("Dark Strokes", self.dark_strokes),
234: ("Ink Outlines", self.ink_outlines),
235: ("Spatter Effect", self.spatter_effect),
236: ("Sprayed Strokes", self.sprayed_strokes),
237: ("Sumi-e Filter", self.sumi_e_filter),
238: ("Underpainting", self.underpainting),
239: ("Accented Edges", self.accented_edges),
240: ("Bas Relief", self.bas_relief),
241: ("Chalk & Charcoal", self.chalk_charcoal),
242: ("Conté Crayon Filter", self.conte_crayon_filter),
243: ("Graphic Pen", self.graphic_pen),
244: ("Halftone Pattern", self.halftone_pattern),
245: ("Note Paper", self.note_paper),
246: ("Photocopy Effect", self.photocopy_effect),
247: ("Plaster Effect", self.plaster_effect),
248: ("Reticulation", self.reticulation),
249: ("Stamp Filter", self.stamp_filter),
250: ("Water Paper", self.water_paper),
251: ("Clouds Filter", self.clouds_filter),
252: ("Difference Clouds", self.difference_clouds),
253: ("Fibers Filter", self.fibers_filter),
254: ("Lens Flare Filter", self.lens_flare_filter),
255: ("Lighting Effects", self.lighting_effects),
256: ("Render Flames", self.render_flames),
257: ("Tree Filter", self.tree_filter),
258: ("Twirl Effect", self.twirl_effect),
259: ("Wave Distortion", self.wave_distortion),
260: ("Zigzag Effect", self.zigzag_effect),
        }
    
    def load_and_preprocess(self, image_path):
        """Load and preprocess image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image. Check the file path.")
        
        # Convert to rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        return img_rgb, gray
    
    def apply_effect(self, image_path, effect_number):
        """Apply specific artistic effect"""
        if effect_number not in self.effects:
            raise ValueError(f"Effect {effect_number} not available. Choose 1-300.")
        
        img_rgb, gray = self.load_and_preprocess(image_path)
        effect_name, effect_function = self.effects[effect_number]
        
        result = effect_function(img_rgb, gray)
        
        return result
   

    def list_all_effects(self):
        """List all available effects"""
        print("Available Artistic Effects:")
        print("=" * 50)
        
        categories = {
            "Traditional Art (1-19)": range(1, 20),
            "Digital Effects (20-35)": range(20, 36),
            "Anime Styles (36-50)": range(36, 51),
            "Paint Styles (51-65)": range(51, 66),
            "Ghibli Styles (66-80)": range(66, 81)
        }
        
        for category, effect_range in categories.items():
            print(f"\n{category}:")
            for i in effect_range:
                if i in self.effects:
                    name, _ = self.effects[i]
                    print(f"  {i:2d}. {name}")
    
# styles code start here--
    def pencil_sketch(self, img_rgb, gray):
        """Classic pencil sketch"""
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        inverted_blur = 255 - blur
        pencil = cv2.divide(gray, inverted_blur, scale=256)
        
        # Add texture
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(pencil, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def charcoal_drawing(self, img_rgb, gray):
        """Rich charcoal effect"""
        bilateral = cv2.bilateralFilter(gray, 15, 40, 40)
        edges = cv2.Canny(bilateral, 50, 150)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
        edges = 255 - edges
        
        blur = cv2.GaussianBlur(bilateral, (25, 25), 0)
        charcoal = cv2.multiply(edges, blur, scale=1/256.0)
        
        # Add texture
        noise = np.random.randint(0, 30, gray.shape, dtype=np.uint8)
        textured = cv2.addWeighted(charcoal, 0.85, noise, 0.15, 0)
        return np.clip(textured, 0, 255).astype(np.uint8)
    
    def ink_sketch(self, img_rgb, gray):
        """Bold ink drawing"""
        median = cv2.medianBlur(gray, 5)
        laplacian = cv2.Laplacian(median, cv2.CV_64F, ksize=3)
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(laplacian)
        
        _, thresh = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
        ink = 255 - thresh
        
        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
        return cleaned
    
    def oil_painting(self, img_rgb, gray):
        """Improved oil painting effect"""
        
        # Step 1: Apply stronger bilateral filtering (edge-preserving smoothing)
        oil = cv2.bilateralFilter(img_rgb, 15, 75, 75)
        oil = cv2.bilateralFilter(oil, 15, 75, 75)
        oil = cv2.bilateralFilter(oil, 15, 75, 75)  # Triple pass for smoothness

        # Step 2: Simulate brush strokes with anisotropic blur kernel
        kernel_diag = np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]], np.float32) / 5
        brush_texture = cv2.filter2D(oil, -1, kernel_diag)

        # Step 3: Color enhancement (more vivid paint look)
        enhanced = cv2.convertScaleAbs(brush_texture, alpha=1.3, beta=15)

        # Step 4: Optional color quantization for paint blob feel
        Z = enhanced.reshape((-1, 3))
        Z = np.float32(Z)
        K = 24  # number of color clusters
        _, labels, centers = cv2.kmeans(Z, K, None,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                        10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()].reshape((img_rgb.shape))

        # Step 5: Sharpening for definition
        sharpen_kernel = np.array([[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]])
        final = cv2.filter2D(quantized, -1, sharpen_kernel)

        return final

    def watercolor(self,img_rgb, gray):
        """Enhanced Realistic Watercolor Painting Effect"""
        # 1. Multi-scale bilateral filtering for layered softness
        base = img_rgb.copy()
        for d, sigma_color, sigma_space in [(9, 50, 50), (7, 30, 30), (5, 15, 15)]:
            base = cv2.bilateralFilter(base, d, sigma_color, sigma_space)

        # 2. Edge-preserving stylization with stronger color flow
        stylized = cv2.stylization(base, sigma_s=150, sigma_r=0.3)

        # 3. Subtle color jitter to mimic pigment variations
        hsv = cv2.cvtColor(stylized, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_variation = (np.random.rand(*hsv[..., 0].shape) - 0.5) * 5  # ±2.5° hue
        sat_variation = (np.random.rand(*hsv[..., 1].shape) - 0.5) * 15  # ±7.5% saturation
        hsv[..., 0] = (hsv[..., 0] + hue_variation) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + sat_variation, 0, 255)
        stylized = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 4. Add watercolor paper texture
        def add_paper_texture(img, intensity=0.2, scale=1.5):
            h, w = img.shape[:2]
            # Generate Perlin noise or use static texture
            noise = (np.random.randn(h, w) * 255).astype(np.uint8)
            noise = cv2.GaussianBlur(noise, (0,0), sigmaX=3)
            noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
            layered = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            textured = layered + (noise.astype(np.float32) / 255.0 - 0.5) * intensity
            textured = np.clip(textured, 0, 1)
            return (textured * 255).astype(np.uint8)

        paper = add_paper_texture(stylized, intensity=0.1)
        textured_color = cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)

        # Blend textured base with color for soft irregularity
        blended = cv2.multiply(stylized.astype(np.float32) / 255,
                                textured_color.astype(np.float32) / 255)
        blended = (blended * 255).astype(np.uint8)

        # 5. Edge enhancement: soft mask from grayscale edges
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        edges = cv2.GaussianBlur(edges, (5,5), sigmaX=0)
        mask = edges.astype(np.float32) / 255.0 * 0.2  # edge strength
        mask = cv2.merge([mask, mask, mask])
        final = np.clip(blended.astype(np.float32) + (mask * 255), 0, 255).astype(np.uint8)

        return final
    
   
    
    def acrylic_paint(self,img_rgb,gray):
        """Enhanced Realistic Acrylic Painting Effect"""
        # Step 1: Edge preserving smoothing (like your base)
        base = cv2.edgePreservingFilter(img_rgb, flags=1, sigma_s=60, sigma_r=0.4)
        
        # Step 2: Boost saturation and slightly adjust hue for vibrance
        hsv = cv2.cvtColor(base, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
        hsv[:, :, 0] = (hsv[:, :, 0] + 5) % 180  # subtle hue shift
        vibrant = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Step 3: Add multi-scale directional brush strokes
        # Smaller brush strokes (fine detail)
        small_brush = directional_blur(vibrant, kernel_size=7, angle=45)
        # Larger brush strokes (broad strokes)
        large_brush = directional_blur(vibrant, kernel_size=15, angle=15)
        # Blend the two for natural effect
        brush_strokes = cv2.addWeighted(small_brush, 0.6, large_brush, 0.4, 0)
        
        # Step 4: Add subtle sharpening to emphasize strokes
        kernel_sharp = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
        sharpened = cv2.filter2D(brush_strokes, -1, kernel_sharp)
        
        # Step 5: Add canvas-like paper texture
        textured = add_paper_texture(sharpened, intensity=10)
        
        # Step 6: Add embossed paint thickness simulation
        gray = cv2.cvtColor(textured, cv2.COLOR_RGB2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        embossed = cv2.addWeighted(textured, 1, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.3, 0)
        
        # Final clipping to valid range
        final = np.clip(embossed, 0, 255).astype(np.uint8)
        
        return final

    def pastel_drawing(self, img_rgb, gray):
        """Realistic Soft Pastel Drawing Effect"""
        # Heavy blur
        blurred = cv2.GaussianBlur(img_rgb, (21, 21), 0)

        # Lower contrast for softness
        pastel = cv2.convertScaleAbs(blurred, alpha=0.7, beta=40)

        # Grainy paper texture
        h, w = gray.shape
        grain = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
        pastel = np.clip(pastel.astype(np.int16) + grain, 0, 255).astype(np.uint8)

        # Light sketch outlines
        final = blend_edges(pastel, gray, edge_strength=0.1)
        return final
    def cross_hatching(self, img_rgb, gray):
        """Cross-hatching drawing technique"""
        h, w = gray.shape
        lines = np.ones((h, w), dtype=np.uint8) * 255
        
        # Create hatching lines based on intensity
        for i in range(0, h, 6):
            for j in range(0, w, 6):
                intensity = gray[i:i+6, j:j+6].mean() / 255.0
                
                if intensity < 0.9:
                    cv2.line(lines, (j, i), (j+6, i+6), 0, 1)
                if intensity < 0.7:
                    cv2.line(lines, (j+6, i), (j, i+6), 0, 1)
                if intensity < 0.5:
                    cv2.line(lines, (j, i+3), (j+6, i+3), 0, 1)
                if intensity < 0.3:
                    cv2.line(lines, (j+3, i), (j+3, i+6), 0, 1)
        
        return lines
    
    def stippling(self, img_rgb, gray):
        """Stippling (dot-based) drawing"""
        h, w = gray.shape
        stippled = np.ones((h, w), dtype=np.uint8) * 255
        
        # Create dots based on intensity
        num_dots = min(h * w // 4, 50000)  # Limit for performance
        
        for _ in range(num_dots):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            intensity = gray[y, x] / 255.0
            
            if np.random.random() > intensity:
                size = max(1, int((1 - intensity) * 3))
                cv2.circle(stippled, (x, y), size, 0, -1)
        
        return stippled
    
    def conte_crayon(self, img_rgb, gray):
        """Conte crayon effect"""
        # Create paper texture
        h, w = gray.shape
        paper = np.random.randint(200, 256, (h, w), dtype=np.uint8)
        
        # Blend with original
        conte = cv2.addWeighted(gray, 0.7, paper, 0.3, 0)
        
        # Add crayon texture
        kernel = np.random.randint(-2, 3, (3, 3)).astype(np.float32)
        kernel = kernel / np.sum(np.abs(kernel)) if np.sum(np.abs(kernel)) != 0 else kernel
        textured = cv2.filter2D(conte, -1, kernel)
        
        return np.clip(textured, 0, 255).astype(np.uint8)
    
    def colored_pencil(self, img_rgb, gray):
        """Colored pencil drawing"""
        # Apply pencil texture to each channel
        result = np.zeros_like(img_rgb)
        
        for i in range(3):
            channel = img_rgb[:,:,i]
            blur = cv2.GaussianBlur(channel, (21, 21), 0)
            inverted_blur = 255 - blur
            pencil = cv2.divide(channel, inverted_blur, scale=256)
            result[:,:,i] = pencil
        
        # Add paper texture
        h, w = gray.shape
        texture = np.random.randint(-5, 6, (h, w, 3), dtype=np.int16)
        result = np.clip(result.astype(np.int16) + texture, 0, 255).astype(np.uint8)
        
        return result
    
    def marker_drawing(self, img_rgb, gray):
        """Marker drawing effect"""
        # Strong color reduction
        marker = cv2.bilateralFilter(img_rgb, 25, 150, 150)
        
        # Quantize colors
        data = marker.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        quantized = quantized.reshape(marker.shape)
        
        # Add slight edges
        edges = cv2.Canny(gray, 50, 100)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        result = cv2.subtract(quantized, edges // 3)
        
        return result
    
    def chalk_drawing(self, img_rgb, gray):
        """Chalk on blackboard effect"""
        # Invert image
        inverted = 255 - gray
        
        # Add chalk texture
        h, w = gray.shape
        chalk_texture = np.random.randint(0, 50, (h, w), dtype=np.uint8)
        chalk = cv2.addWeighted(inverted, 0.7, chalk_texture, 0.3, 0)
        
        # Create blackboard background
        blackboard = np.zeros((h, w), dtype=np.uint8)
        
        # Blend chalk with blackboard
        result = cv2.addWeighted(blackboard, 0.2, chalk, 0.8, 0)
        
        return result
    
    def ballpoint_pen(self, img_rgb, gray):
        """Ballpoint pen drawing"""
        # Create fine lines
        edges = cv2.Canny(gray, 100, 200)
        
        # Add pen texture with cross-hatching
        h, w = gray.shape
        pen_drawing = np.ones((h, w), dtype=np.uint8) * 255
        
        # Fine hatching based on intensity
        for i in range(0, h, 4):
            for j in range(0, w, 4):
                if i < h and j < w:
                    intensity = gray[i, j] / 255.0
                    if intensity < 0.8:
                        cv2.line(pen_drawing, (j, i), (j+3, i+1), 0, 1)
                    if intensity < 0.6:
                        cv2.line(pen_drawing, (j+1, i), (j+2, i+3), 0, 1)
        
        # Combine with edges
        result = cv2.bitwise_and(pen_drawing, 255 - edges)
        return result
    
    def scratchboard(self, img_rgb, gray):
        """Scratchboard engraving effect"""
        # Start with black background
        scratch = np.zeros_like(gray)
        
        # Create white scratches based on edges and highlights
        edges = cv2.Canny(gray, 50, 150)
        
        # Add fine line texture
        h, w = gray.shape
        for i in range(0, h, 3):
            for j in range(0, w, 3):
                if gray[i, j] > 128:  # Light areas become white scratches
                    cv2.line(scratch, (j, i), (j+2, i), 255, 1)
                    if gray[i, j] > 180:
                        cv2.line(scratch, (j, i), (j, i+2), 255, 1)
        
        # Combine with edges
        result = cv2.bitwise_or(scratch, edges)
        return result
    
    def lithograph(self, img_rgb, gray):
        """Lithograph print effect"""
        # Create grainy texture
        h, w = gray.shape
        grain = np.random.randint(0, 30, (h, w), dtype=np.uint8)
        
        # Apply grain
        litho = cv2.addWeighted(gray, 0.8, grain, 0.2, 0)
        
        # Increase contrast
        litho = cv2.convertScaleAbs(litho, alpha=1.5, beta=-50)
        
        # Add print lines
        for i in range(0, h, 2):
            litho[i, :] = litho[i, :] * 0.9
        
        return np.clip(litho, 0, 255).astype(np.uint8)
    
    def woodcut_print(self, img_rgb, gray):
        """Woodcut print effect"""
        # High contrast
        _, woodcut = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        # Add wood grain texture
        h, w = gray.shape
        for i in range(0, h, 5):
            noise = np.random.randint(-20, 21, w)
            if i < h:
                woodcut[i, :] = np.clip(woodcut[i, :].astype(np.int16) + noise, 0, 255)
        
        return woodcut.astype(np.uint8)
    
    def linocut(self, img_rgb, gray):
        """Linocut print effect"""
        # Strong threshold
        _, lino = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        # Add lino texture
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        textured = cv2.filter2D(lino, -1, kernel)
        
        return np.clip(textured + 128, 0, 255).astype(np.uint8)
    
    def etching(self, img_rgb, gray):
        """Etching effect"""
        # Fine line work
        edges = cv2.Canny(gray, 30, 100)
        
        # Add etching lines
        h, w = gray.shape
        etched = np.ones((h, w), dtype=np.uint8) * 255
        
        # Create fine parallel lines for shading
        for i in range(0, h, 2):
            intensity = np.mean(gray[i, :]) / 255.0
            if intensity < 0.7:
                etched[i, :] = etched[i, :] * intensity
        
        # Combine with edges
        result = cv2.bitwise_and(etched, 255 - edges)
        return result
    
    def night_vision(self, img_rgb, gray):
        """Night vision effect"""
        # Convert to green tint
        night = np.zeros_like(img_rgb)
        night[:,:,1] = gray  # Green channel
        
        # Add noise
        h, w = gray.shape
        noise = np.random.randint(0, 50, (h, w), dtype=np.uint8)
        night[:,:,1] = cv2.addWeighted(night[:,:,1], 0.8, noise, 0.2, 0)
        
        # Add scanlines
        for i in range(0, h, 3):
            if i < h:
                night[i, :, 1] = night[i, :, 1] * 0.7
        
        return night
    
    def thermal_vision(self, img_rgb, gray):
        """Thermal imaging effect"""
        # Create heat map
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
        
        # Add thermal noise
        h, w = gray.shape
        noise = np.random.randint(-10, 11, (h, w, 3), dtype=np.int16)
        thermal = np.clip(thermal.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return thermal
    
    def xray_effect(self, img_rgb, gray):
        """X-ray effect"""
        # Invert and adjust
        xray = 255 - gray
        xray = cv2.convertScaleAbs(xray, alpha=1.2, beta=30)
        
        # Add blue tint
        xray_colored = np.zeros_like(img_rgb)
        xray_colored[:,:,2] = xray  # Blue channel
        xray_colored[:,:,0] = xray // 3  # Slight red
        
        return xray_colored
    
    def neon_glow(self, img_rgb, gray):
        """Neon glow effect"""
        # Find edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Create glow
        glow = cv2.GaussianBlur(edges, (21, 21), 0)
        
        # Create neon colors
        neon = np.zeros_like(img_rgb)
        neon[:,:,0] = glow  # Red
        neon[:,:,1] = edges  # Green (sharp edges)
        neon[:,:,2] = glow // 2  # Blue
        
        # Add to dark background
        dark_bg = img_rgb // 4
        result = cv2.addWeighted(dark_bg, 0.3, neon, 0.7, 0)
        
        return result
    
    def cyberpunk(self, img_rgb, gray):
        """Cyberpunk aesthetic"""
        # High contrast
        cyber = cv2.convertScaleAbs(img_rgb, alpha=1.5, beta=-30)
        
        # Add cyan and magenta tint
        hsv = cv2.cvtColor(cyber, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.5)  # Increase saturation
        cyber = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add scan lines
        h, w = gray.shape
        for i in range(0, h, 4):
            if i < h:
                cyber[i, :] = cyber[i, :] * 0.8
        
        # Add digital noise
        noise = np.random.randint(-15, 16, img_rgb.shape, dtype=np.int16)
        cyber = np.clip(cyber.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return cyber
    
    def vintage_sepia(self, img_rgb, gray):
        """Vintage sepia tone"""
        # Sepia transformation matrix
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])
        
        sepia = img_rgb.dot(sepia_filter.T)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        
        # Add vintage texture
        h, w = gray.shape
        vintage_texture = np.random.randint(-10, 11, (h, w, 3), dtype=np.int16)
        sepia = np.clip(sepia.astype(np.int16) + vintage_texture, 0, 255).astype(np.uint8)
        
        # Vignette effect
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        vignette = 1 - (dist_from_center / max_dist) * 0.3
        
        for i in range(3):
            sepia[:,:,i] = (sepia[:,:,i] * vignette).astype(np.uint8)
        
        return sepia
    
    def blueprint(self, img_rgb, gray):
        """Blueprint technical drawing"""
        # Invert and make blue
        inverted = 255 - gray
        
        # Create blueprint background
        blueprint = np.zeros_like(img_rgb)
        blueprint[:,:,2] = 100  # Blue background
        blueprint[:,:,0] = inverted  # White lines in red channel
        blueprint[:,:,1] = inverted  # White lines in green channel
        blueprint[:,:,2] = np.maximum(blueprint[:,:,2], inverted)  # White lines in blue channel
        
        # Add grid lines
        h, w = gray.shape
        for i in range(0, h, 20):
            if i < h:
                blueprint[i, :, :] = np.maximum(blueprint[i, :, :], 50)
        for j in range(0, w, 20):
            if j < w:
                blueprint[:, j, :] = np.maximum(blueprint[:, j, :], 50)
        
        return blueprint
    
    def pop_art(self, img_rgb, gray):
        """Pop art style"""
        # Posterize colors
        pop = img_rgb // 64 * 64  # Reduce to 4 levels per channel
        
        # Enhance saturation
        hsv = cv2.cvtColor(pop, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = np.minimum(hsv[:,:,1] * 2, 255)  # Double saturation
        pop = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add strong edges
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8))
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.addWeighted(pop, 0.9, edges_colored, 0.1, 0)
        return result
    
    def comic_book(self, img_rgb, gray):
        """Comic book style"""
        # Strong bilateral filter
        comic = cv2.bilateralFilter(img_rgb, 25, 200, 200)
        
        # Quantize colors
        data = comic.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
        _, labels, centers = cv2.kmeans(data, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        quantized = quantized.reshape(comic.shape)
        
        # Add thick black edges
        edges = cv2.Canny(gray, 50, 100)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(quantized, edges_colored)
        return result
    
    def anime_style(self, img_rgb, gray):
        """Basic anime style"""
        # Smooth skin/surfaces
        anime = cv2.bilateralFilter(img_rgb, 20, 100, 100)
        
        # Sharp edges for anime look
        edges = cv2.Canny(gray, 30, 80)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8))
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Enhance colors
        hsv = cv2.cvtColor(anime, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.4)  # Increase saturation
        anime = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        result = cv2.subtract(anime, edges_colored // 2)
        return result
    
    def abstract_art(self, img_rgb, gray):
        """Abstract art effect"""
        # Create abstract shapes
        h, w = gray.shape
        abstract = np.copy(img_rgb)
        
        # Random color blocks
        for _ in range(20):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = min(x1 + np.random.randint(20, 100), w), min(y1 + np.random.randint(20, 100), h)
            color = np.random.randint(0, 256, 3)
            cv2.rectangle(abstract, (x1, y1), (x2, y2), color.tolist(), -1)
        
        # Blend with original
        result = cv2.addWeighted(img_rgb, 0.6, abstract, 0.4, 0)
        
        # Add texture
        noise = np.random.randint(-30, 31, img_rgb.shape, dtype=np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return result
    
    def impressionist(self, img_rgb, gray):
        """Advanced and more realistic Impressionist painting style"""

        # Step 1: Apply Gaussian Blur for soft base
        blurred = cv2.GaussianBlur(img_rgb, (9, 9), 0)

        # Step 2: Bilateral Filter to preserve edges
        smoothed = cv2.bilateralFilter(blurred, 15, 80, 80)

        # Step 3: Color Quantization (k-means for posterized color)
        Z = img_rgb.reshape((-1, 3))
        Z = np.float32(Z)
        K = 10
        _, labels, centers = cv2.kmeans(Z, K, None,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                        10, cv2.KMEANS_RANDOM_CENTERS)
        quantized = centers[labels.flatten()].reshape(img_rgb.shape).astype(np.uint8)

        # Step 4: Edge painting (detect edges and blend)
        edges = cv2.Canny(gray, 80, 120)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        quantized[edges > 0] = [0, 0, 0]  # black strokes

        # Step 5: Simulate directional brush strokes with motion blur
        kernel_motion_blur = np.zeros((9, 9))
        kernel_motion_blur[4, :] = np.ones(9)  # horizontal brush stroke
        kernel_motion_blur /= 9
        brushed = cv2.filter2D(quantized, -1, kernel_motion_blur)

        # Step 6: Add canvas texture (using Perlin-style noise or emboss)
        noise = np.random.normal(0, 5, img_rgb.shape).astype(np.uint8)
        textured = cv2.add(brushed, noise)

        # Step 7: Emboss filter for raised paint effect
        kernel_emboss = np.array([[ -2, -1,  0],
                                [ -1,  1,  1],
                                [  0,  1,  2]])
        embossed = cv2.filter2D(textured, -1, kernel_emboss)

        # Step 8: Final brightening and tone
        final = cv2.convertScaleAbs(embossed, alpha=1.2, beta=30)

        return final

    
    def pointillism(self, img_rgb, gray):
        """Pointillism (Seurat style)"""
        h, w = gray.shape
        pointillism = np.ones_like(img_rgb) * 255
        
        # Create dots of color
        num_dots = min(h * w // 8, 30000)  # Limit for performance
        
        for _ in range(num_dots):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            color = img_rgb[y, x]
            
            # Vary dot size based on local contrast
            local_contrast = np.std(gray[max(0,y-5):y+5, max(0,x-5):x+5])
            dot_size = max(2, int(local_contrast / 10))
            
            cv2.circle(pointillism, (x, y), dot_size, color.tolist(), -1)
        
        return pointillism
    
    def expressionist(self, img_rgb, gray):
        """Expressionist style"""
        # Distort and enhance emotions
        expressionist = cv2.bilateralFilter(img_rgb, 10, 150, 150)
        
        # Extreme color enhancement
        hsv = cv2.cvtColor(expressionist, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = np.minimum(hsv[:,:,1] * 2.5, 255)  # Extreme saturation
        expressionist = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # High contrast
        expressionist = cv2.convertScaleAbs(expressionist, alpha=1.8, beta=-50)
        
        # Add emotional texture
        h, w = gray.shape
        emotion_texture = np.random.randint(-40, 41, (h, w, 3), dtype=np.int16)
        result = np.clip(expressionist.astype(np.int16) + emotion_texture, 0, 255).astype(np.uint8)
        
        return result
    
    def cubist(self, img_rgb, gray):
        """Cubist fragmentation"""
        h, w = gray.shape
        cubist = np.copy(img_rgb)
        
        # Create geometric fragments
        for _ in range(15):
            # Random triangle
            pts = np.array([[np.random.randint(0, w), np.random.randint(0, h)] for _ in range(3)], np.int32)
            
            # Sample color from original image
            center_x, center_y = np.mean(pts, axis=0).astype(int)
            center_x, center_y = min(max(center_x, 0), w-1), min(max(center_y, 0), h-1)
            color = img_rgb[center_y, center_x]
            
            cv2.fillPoly(cubist, [pts], color.tolist())
        
        # Add geometric lines
        for _ in range(20):
            pt1 = (np.random.randint(0, w), np.random.randint(0, h))
            pt2 = (np.random.randint(0, w), np.random.randint(0, h))
            cv2.line(cubist, pt1, pt2, (0, 0, 0), 2)
        
        return cubist
    
    def art_nouveau(self, img_rgb, gray):
        """Art Nouveau organic style"""
        # Smooth organic curves
        nouveau = cv2.bilateralFilter(img_rgb, 20, 100, 100)
        
        # Enhance flowing lines
        edges = cv2.Canny(gray, 30, 70)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        
        # Organic color palette
        hsv = cv2.cvtColor(nouveau, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = (hsv[:,:,0] + 30) % 180  # Shift toward greens/blues
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.3)
        nouveau = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Blend with flowing lines
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        result = cv2.subtract(nouveau, edges_colored // 3)
        
        return result
    
    # NEW ANIME STYLES (36-50)
    def shoujo_anime(self, img_rgb, gray):
        
        """Improved Shoujo anime style - soft, dreamy, romantic"""

        
        # Step 1: Smoother skin with double bilateral filter
        smoothed = cv2.bilateralFilter(img_rgb, 25, 120, 120)
        smoothed = cv2.bilateralFilter(smoothed, 15, 90, 90)

        # Step 2: Slight glow with Gaussian blur blend
        glow = cv2.GaussianBlur(smoothed, (9, 9), 0)
        shoujo_base = cv2.addWeighted(smoothed, 0.8, glow, 0.2, 0)

        # Step 3: Bright pastel tone adjustment
        pastel = cv2.convertScaleAbs(shoujo_base, alpha=1.05, beta=30)

        # Step 4: Light sketch-like edges (thin outlines)
        edges = cv2.Canny(gray, 10, 50)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        pastel = cv2.subtract(pastel, edges_colored // 3)

        # Step 5: Add subtle highlight/sparkles
        h, w = gray.shape
        sparkles = np.zeros_like(img_rgb)
        for _ in range(100):  # More sparkles
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            color = tuple(np.random.randint(200, 256, size=3).tolist())
            cv2.circle(sparkles, (x, y), 1, color, -1)
        pastel = cv2.addWeighted(pastel, 0.95, sparkles, 0.05, 0)

        # Step 6: Add a soft radial light glow (from top-center)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 5), int(min(h, w) * 0.5), 255, -1)
        mask = cv2.GaussianBlur(mask, (101, 101), 0)
        light = cv2.merge([mask]*3)
        light = cv2.normalize(light.astype(np.float32), None, 0, 64, cv2.NORM_MINMAX).astype(np.uint8)
        pastel = cv2.add(pastel, light)

        return pastel

        
    def shounen_anime(self, img_rgb, gray):
        """Shounen anime style - bold and dynamic"""
        # Sharp contrast
        shounen = cv2.convertScaleAbs(img_rgb, alpha=1.3, beta=-20)
        
        # Strong bilateral filter but keep edges
        shounen = cv2.bilateralFilter(shounen, 15, 120, 120)
        
        # Bold edges
        edges = cv2.Canny(gray, 50, 120)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8))
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Vibrant colors
        hsv = cv2.cvtColor(shounen, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.5)
        shounen = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        result = cv2.subtract(shounen, edges_colored)
        return result
    
    def kawaii_style(self, img_rgb, gray):
        """Kawaii cute style"""
        # Super soft and bright
        kawaii = cv2.bilateralFilter(img_rgb, 20, 100, 100)
        kawaii = cv2.convertScaleAbs(kawaii, alpha=1.2, beta=30)
        
        # Pastel color adjustment
        hsv = cv2.cvtColor(kawaii, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = hsv[:,:,1] * 0.8  # Reduce saturation for pastel
        hsv[:,:,2] = np.minimum(hsv[:,:,2] * 1.3, 255)  # Increase brightness
        kawaii = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Very soft edges
        edges = cv2.Canny(gray, 10, 40)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(kawaii, edges_colored // 6)
        return result
    
    def chibi_style(self, img_rgb, gray):
        """Chibi style - simplified and cute"""
        # Heavy simplification
        chibi = cv2.bilateralFilter(img_rgb, 30, 200, 200)
        
        # Color quantization for simple look
        data = chibi.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
        _, labels, centers = cv2.kmeans(data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        chibi = quantized.reshape(chibi.shape)
        
        # Simple thick edges
        edges = cv2.Canny(gray, 30, 90)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8))
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(chibi, edges_colored)
        return result
    
    def cel_shading(self, img_rgb, gray):
        """Traditional cel animation shading"""
        # Flat color regions
        cel = cv2.bilateralFilter(img_rgb, 20, 150, 150)
        
        # Posterize for flat shading
        cel = (cel // 32) * 32  # Reduce to specific color levels
        
        # Clean black outlines
        edges = cv2.Canny(gray, 40, 100)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8))
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(cel, edges_colored)
        return result
    
    def manga_bw(self, img_rgb, gray):
        """Manga black and white style"""
        # High contrast black and white
        manga = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)
        
        # Add screen tones (halftone patterns)
        h, w = gray.shape
        for i in range(0, h, 4):
            for j in range(0, w, 4):
                if 50 < gray[i, j] < 150:  # Mid-tones get dot pattern
                    cv2.circle(manga, (j, i), 1, 0, -1)
        
        return manga
    
    def anime_screenshot(self, img_rgb, gray):
        """Modern anime screenshot style"""
        # Professional anime look
        anime = cv2.bilateralFilter(img_rgb, 15, 100, 100)
        
        # Subtle color adjustment
        hsv = cv2.cvtColor(anime, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.2)
        anime = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Crisp but not thick edges
        edges = cv2.Canny(gray, 40, 90)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(anime, edges_colored // 2)
        return result
    
    def visual_novel(self, img_rgb, gray):
        """Visual novel character style"""
        # Very smooth rendering
        vn = cv2.bilateralFilter(img_rgb, 25, 120, 120)
        
        # Soft lighting effect
        vn = cv2.convertScaleAbs(vn, alpha=1.1, beta=15)
        
        # Minimal edges
        edges = cv2.Canny(gray, 25, 75)
        edges = cv2.GaussianBlur(edges, (2, 2), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(vn, edges_colored // 3)
        return result
    
    def moe_style(self, img_rgb, gray):
        """Moe anime style"""
        # Ultra soft and appealing
        moe = cv2.bilateralFilter(img_rgb, 30, 150, 150)
        
        # Enhance warm colors
        hsv = cv2.cvtColor(moe, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = (hsv[:,:,0] - 15) % 180  # Shift toward warmer hues
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.1)
        moe = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Very soft edges
        edges = cv2.Canny(gray, 15, 50)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(moe, edges_colored // 5)
        return result
    
    def tsundere_aesthetic(self, img_rgb, gray):
        """Tsundere character aesthetic"""
        # Slightly cooler color palette
        tsun = cv2.bilateralFilter(img_rgb, 20, 100, 100)
        
        hsv = cv2.cvtColor(tsun, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = (hsv[:,:,0] + 20) % 180  # Shift toward cooler hues
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.2)
        tsun = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Medium contrast edges
        edges = cv2.Canny(gray, 35, 85)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(tsun, edges_colored // 2)
        return result
    
    def dark_anime(self, img_rgb, gray):
        """Dark anime style"""
        # Darker overall tone
        dark = cv2.convertScaleAbs(img_rgb, alpha=0.8, beta=-30)
        dark = cv2.bilateralFilter(dark, 15, 100, 100)
        
        # Desaturate slightly
        hsv = cv2.cvtColor(dark, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = hsv[:,:,1] * 0.9
        dark = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Strong edges for dramatic effect
        edges = cv2.Canny(gray, 60, 120)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8))
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(dark, edges_colored)
        return result
    
    def magical_girl(self, img_rgb, gray):
        """Magical girl anime style"""
        # Bright and magical
        magical = cv2.bilateralFilter(img_rgb, 20, 120, 120)
        magical = cv2.convertScaleAbs(magical, alpha=1.3, beta=20)
        
        # Enhance magical colors (pinks, purples)
        hsv = cv2.cvtColor(magical, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.4)
        magical = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add sparkle effects
        h, w = gray.shape
        sparkles = np.zeros_like(img_rgb)
        for _ in range(100):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            if np.random.random() > 0.7:  # Random sparkles
                cv2.circle(sparkles, (x, y), np.random.randint(1, 4), (255, 255, 255), -1)
        
        # Soft edges
        edges = cv2.Canny(gray, 30, 80)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.addWeighted(magical, 0.9, sparkles, 0.1, 0)
        result = cv2.subtract(result, edges_colored // 3)
        
        return result
    
    def mecha_anime(self, img_rgb, gray):
        """Mecha anime style"""
        # Sharp mechanical look
        mecha = cv2.bilateralFilter(img_rgb, 10, 150, 150)
        mecha = cv2.convertScaleAbs(mecha, alpha=1.4, beta=-20)
        
        # Cooler color palette
        hsv = cv2.cvtColor(mecha, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = (hsv[:,:,0] + 30) % 180  # Shift toward blues
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.3)
        mecha = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Very sharp edges
        edges = cv2.Canny(gray, 70, 140)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8))
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(mecha, edges_colored)
        return result
    
    def slice_of_life(self, img_rgb, gray):
        """Slice of life anime style"""
        # Natural and comfortable
        sol = cv2.bilateralFilter(img_rgb, 18, 90, 90)
        
        # Warm, natural colors
        hsv = cv2.cvtColor(sol, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = (hsv[:,:,0] - 10) % 180  # Slightly warmer
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.1)
        sol = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Gentle edges
        edges = cv2.Canny(gray, 25, 70)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(sol, edges_colored // 3)
        return result
    
    def anime_portrait(self, img_rgb, gray):
        """Enhanced Anime Portrait Style - with eye sparkle and blush"""

        # Load Haar Cascade for face and eye detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Step 1: Soft smoothing
        smoothed = cv2.bilateralFilter(img_rgb, 25, 150, 150)
        smoothed = cv2.bilateralFilter(smoothed, 15, 100, 100)

        # Step 2: Pastel coloring
        pastel = cv2.convertScaleAbs(smoothed, alpha=1.15, beta=25)

        # Step 3: Anime-like outlines
        edges = cv2.Canny(gray, 40, 100)
        edges = cv2.dilate(edges, None, iterations=1)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        outlined = cv2.subtract(pastel, edges_colored // 2)

        # Step 4: Detect face for blush and eyes for sparkles
        result = outlined.copy()
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Blush on cheeks
            cx = x + w // 4
            cy = y + h // 2
            cv2.circle(result, (cx, cy), 20, (180, 128, 128), -1)
            cv2.circle(result, (x + 3 * w // 4, cy), 20, (180, 128, 128), -1)

            # Detect eyes in the face region
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = result[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                sparkle_x = x + ex + ew // 2
                sparkle_y = y + ey + eh // 2
                cv2.circle(result, (sparkle_x, sparkle_y), 3, (255, 255, 255), -1)
                cv2.circle(result, (sparkle_x + 2, sparkle_y - 2), 2, (255, 255, 255), -1)

        # Step 5: Gaussian blur on blush for natural look
        result = cv2.GaussianBlur(result, (3, 3), 0)

        # Step 6: Sharpen
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        result = cv2.filter2D(result, -1, sharpen_kernel)

        return result

        # NEW PAINT STYLES (51-65)
    def fresco_painting(self, img_rgb, gray):
        """Italian fresco painting style"""
        # Aged and weathered look
        h, w = gray.shape

    # Step 1: Desaturate & fade colors (earthy palette)
        fresco = cv2.bilateralFilter(img_rgb, 15, 75, 75)
        hsv = cv2.cvtColor(fresco, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.5     # faded saturation
        hsv[:, :, 2] = hsv[:, :, 2] * 0.75    # faded brightness
        fresco = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Step 2: Add wall-like plaster texture
        texture = np.random.normal(loc=0, scale=12, size=(h, w, 1)).astype(np.float32)
        texture = cv2.GaussianBlur(texture, (13, 13), 0)
        fresco = np.clip(fresco + texture, 0, 255).astype(np.uint8)

        # Step 3: Add realistic vignette (darkened edges)
        Y, X = np.indices((h, w))
        cx, cy = w / 2, h / 2
        vignette_mask = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * (0.55 * w)**2))
        fresco = (fresco.astype(np.float32) * vignette_mask[..., None]).astype(np.uint8)

        # Step 4: Overlay fine cracks using random walk lines
        crack_layer = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(30):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            for _ in range(30):
                dx = np.random.randint(-3, 4)
                dy = np.random.randint(-3, 4)
                nx, ny = np.clip(x + dx, 0, w - 1), np.clip(y + dy, 0, h - 1)
                cv2.line(crack_layer, (x, y), (nx, ny), (50, 40, 30), 1, cv2.LINE_AA)
                x, y = nx, ny

        # Blend cracks subtly
        fresco = cv2.addWeighted(fresco, 0.97, crack_layer, 0.03, 0)

        # Step 5: Simulate pigment blotches (water damage stains)
        blotch = np.zeros_like(fresco, dtype=np.uint8)
        for _ in range(5):
            radius = np.random.randint(20, 60)
            center = (np.random.randint(0, w), np.random.randint(0, h))
            color = (np.random.randint(70, 120), np.random.randint(60, 100), np.random.randint(50, 90))
            cv2.circle(blotch, center, radius, color, -1)
        blotch = cv2.GaussianBlur(blotch, (25, 25), 10)
        fresco = cv2.addWeighted(fresco, 0.95, blotch, 0.05, 0)

        return fresco
        
    def tempera_paint(self, img_rgb, gray):
        """Tempera painting technique"""
        # Fine, detailed strokes
        tempera = cv2.bilateralFilter(img_rgb, 10, 120, 120)
        
        # Add fine brush texture
        kernel = np.array([[-1, -1, -1], [-1, 12, -1], [-1, -1, -1]]) / 4
        textured = cv2.filter2D(tempera, -1, kernel)
        
        # Slightly matte finish
        matte = cv2.convertScaleAbs(textured, alpha=0.95, beta=5)
        
        return matte
    
    def gouache_style(self, img_rgb, gray):
        """Gouache painting style"""
        # Opaque, matte appearance
        gouache = cv2.bilateralFilter(img_rgb, 15, 100, 100)
        
        # Reduce brightness for matte look
        gouache = cv2.convertScaleAbs(gouache, alpha=0.9, beta=-10)
        
        # Add paper texture
        h, w = gray.shape
        paper_texture = np.random.randint(-15, 16, (h, w, 3), dtype=np.int16)
        gouache = np.clip(gouache.astype(np.int16) + paper_texture, 0, 255).astype(np.uint8)
        
        return gouache
    
    def encaustic_wax(self, img_rgb, gray):
        """Encaustic wax painting"""
        # Smooth, flowing appearance
        encaustic = cv2.GaussianBlur(img_rgb, (7, 7), 0)
        encaustic = cv2.bilateralFilter(encaustic, 20, 100, 100)
        
        # Add wax-like sheen
        sheen = cv2.convertScaleAbs(encaustic, alpha=1.1, beta=15)
        
        # Blend colors more
        kernel = np.ones((5, 5), np.float32) / 25
        blended = cv2.filter2D(sheen, -1, kernel)
        
        return blended
    
    def chinese_ink_wash(self, img_rgb, gray):
        """Chinese ink wash painting"""
        # Convert to grayscale with subtle color
        ink_wash = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Add subtle sepia tone
        sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
        ink_wash = ink_wash.dot(sepia_kernel.T)
        ink_wash = np.clip(ink_wash, 0, 255).astype(np.uint8)
        
        # Add water effects
        water_blur = cv2.GaussianBlur(ink_wash, (15, 15), 0)
        result = cv2.addWeighted(ink_wash, 0.7, water_blur, 0.3, 0)
        
        return result
    
    def japanese_sumi_e(self, img_rgb, gray):
        """Japanese Sumi-e painting"""
        # Minimalist black ink
        sumi = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)
        
        # Add brush stroke texture
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7))  # Brush-like
        sumi = cv2.morphologyEx(sumi, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to RGB with ink color
        sumi_colored = cv2.cvtColor(sumi, cv2.COLOR_GRAY2RGB)
        
        # Add subtle ink variations
        h, w = gray.shape
        ink_variation = np.random.randint(-20, 21, (h, w, 3), dtype=np.int16)
        result = np.clip(sumi_colored.astype(np.int16) + ink_variation, 0, 255).astype(np.uint8)
        
        return result
    
    def dutch_masters(self, img_rgb, gray):
        """Dutch Masters painting style"""
        # Rich, deep colors
        dutch = cv2.bilateralFilter(img_rgb, 15, 100, 100)
        
        # Enhance contrast and depth
        dutch = cv2.convertScaleAbs(dutch, alpha=1.3, beta=-30)
        
        # Add golden hour lighting
        hsv = cv2.cvtColor(dutch, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = (hsv[:,:,0] - 20) % 180  # Warmer hues
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.2)
        dutch = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add vignette for classical look
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        vignette = 1 - (dist_from_center / max_dist) * 0.4
        
        for i in range(3):
            dutch[:,:,i] = (dutch[:,:,i] * vignette).astype(np.uint8)
        
        return dutch
    
    def fauvism(self, img_rgb, gray):
        """Fauvism wild colors"""
        # Extreme color saturation
        fauv = cv2.bilateralFilter(img_rgb, 20, 150, 150)
        
        hsv = cv2.cvtColor(fauv, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = np.minimum(hsv[:,:,1] * 3, 255)  # Triple saturation
        hsv[:,:,0] = (hsv[:,:,0] + np.random.randint(-30, 31, hsv.shape[:2])) % 180  # Wild hue shifts
        fauv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Bold brush strokes
        kernel = np.array([[0, -2, 0], [-2, 10, -2], [0, -2, 0]]) / 2
        textured = cv2.filter2D(fauv, -1, kernel)
        
        return np.clip(textured, 0, 255).astype(np.uint8)
    
    def neo_impressionism(self, img_rgb, gray):
        """Neo-Impressionism divisionism"""
        # Systematic color separation
        neo = cv2.bilateralFilter(img_rgb, 10, 80, 80)
        
        # Separate into color components
        h, w = gray.shape
        result = np.zeros_like(img_rgb)
        
        # Create systematic dots of pure color
        for i in range(0, h, 4):
            for j in range(0, w, 4):
                if i < h and j < w:
                    color = neo[i, j]
                    # Place color dots systematically
                    cv2.circle(result, (j, i), 2, color.tolist(), -1)
                    cv2.circle(result, (j+2, i), 1, (color * 0.8).astype(int).tolist(), -1)
        
        return result
    
    def abstract_expressionism(self, img_rgb, gray):
        """Abstract Expressionism style"""
        # Large gestural strokes
        abstract = cv2.bilateralFilter(img_rgb, 25, 200, 200)
        
        # Create large color fields
        h, w = gray.shape
        for _ in range(10):
            # Random large brush strokes
            center = (np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4))
            axes = (np.random.randint(50, 150), np.random.randint(50, 150))
            angle = np.random.randint(0, 180)
            color = np.random.randint(0, 256, 3).tolist()
            
            cv2.ellipse(abstract, center, axes, angle, 0, 360, color, -1)
        
        # Blend with original
        result = cv2.addWeighted(img_rgb, 0.6, abstract, 0.4, 0)
        
        return result
    
    def color_field_painting(self, img_rgb, gray):
        """Color Field painting style"""
        # Large areas of flat color
        field = cv2.bilateralFilter(img_rgb, 30, 200, 200)
        
        # Posterize heavily
        field = (field // 64) * 64
        
        # Create color fields
        h, w = gray.shape
        regions = np.zeros_like(img_rgb)
        
        # Divide into regions
        for i in range(0, h, h//3):
            for j in range(0, w, w//3):
                avg_color = np.mean(field[i:i+h//3, j:j+w//3], axis=(0,1)).astype(int)
                regions[i:i+h//3, j:j+w//3] = avg_color
        
        # Blend slightly with original
        result = cv2.addWeighted(field, 0.3, regions, 0.7, 0)
        
        return result
    
    def plein_air(self, img_rgb, gray):
        """Plein air outdoor painting"""
        # Natural lighting and colors
        plein = cv2.bilateralFilter(img_rgb, 15, 100, 100)
        
        # Enhance natural light
        plein = cv2.convertScaleAbs(plein, alpha=1.1, beta=20)
        
        # Add atmospheric perspective
        h, w = gray.shape
        atmosphere = np.ones_like(img_rgb) * [200, 210, 255]  # Sky color
        
        # Blend with distance effect
        for i in range(h):
            blend_factor = i / h * 0.2  # More atmosphere with distance
            plein[i, :] = cv2.addWeighted(plein[i, :], 1-blend_factor, 
                                         atmosphere[i, :], blend_factor, 0)
        
        return plein
    
    def baroque_painting(self, img_rgb, gray):
        """Baroque dramatic lighting"""
        # Strong chiaroscuro effect
        baroque = cv2.bilateralFilter(img_rgb, 20, 120, 120)
        
        # Enhance contrast dramatically
        baroque = cv2.convertScaleAbs(baroque, alpha=1.8, beta=-80)
        
        # Add golden tones
        hsv = cv2.cvtColor(baroque, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = (hsv[:,:,0] - 15) % 180  # Golden hues
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.3)
        baroque = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Strong directional lighting
        h, w = gray.shape
        light_source = (w//4, h//4)  # Top-left light
        Y, X = np.ogrid[:h, :w]
        dist_from_light = np.sqrt((X - light_source[0])**2 + (Y - light_source[1])**2)
        max_dist = np.sqrt(w**2 + h**2)
        lighting = 1.2 - (dist_from_light / max_dist) * 0.6
        
        for i in range(3):
            baroque[:,:,i] = np.clip(baroque[:,:,i] * lighting, 0, 255).astype(np.uint8)
        
        return baroque
    
    def renaissance_style(self, img_rgb, gray):
        """Renaissance painting technique"""
        # Sfumato technique - soft transitions
        renaissance = cv2.bilateralFilter(img_rgb, 25, 150, 150)
        renaissance = cv2.GaussianBlur(renaissance, (5, 5), 0)
        
        # Classical color palette
        hsv = cv2.cvtColor(renaissance, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = hsv[:,:,1] * 0.8  # Reduce saturation
        hsv[:,:,2] = hsv[:,:,2] * 0.9  # Slightly darker
        renaissance = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add aged appearance
        h, w = gray.shape
        age_effect = np.random.randint(-10, 11, (h, w, 3), dtype=np.int16)
        renaissance = np.clip(renaissance.astype(np.int16) + age_effect, 0, 255).astype(np.uint8)
        
        return renaissance
    
    def modern_abstract(self, img_rgb, gray):
        """Modern abstract style"""
        # Geometric abstraction
        abstract = np.copy(img_rgb)
        h, w = gray.shape
        
        # Create geometric shapes
        for _ in range(20):
            shape_type = np.random.choice(['rectangle', 'circle', 'triangle'])
            color = np.random.randint(0, 256, 3).tolist()
            
            if shape_type == 'rectangle':
                x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
                x2, y2 = min(x1 + np.random.randint(20, 100), w), min(y1 + np.random.randint(20, 100), h)
                cv2.rectangle(abstract, (x1, y1), (x2, y2), color, -1)
            elif shape_type == 'circle':
                center = (np.random.randint(0, w), np.random.randint(0, h))
                radius = np.random.randint(10, 50)
                cv2.circle(abstract, center, radius, color, -1)
            else:  # triangle
                pts = np.array([[np.random.randint(0, w), np.random.randint(0, h)] for _ in range(3)], np.int32)
                cv2.fillPoly(abstract, [pts], color)
        
        # Blend with original
        result = cv2.addWeighted(img_rgb, 0.4, abstract, 0.6, 0)
        
        return result
    
    # STUDIO GHIBLI STYLES (66-80)
    def ghibli_landscape(self, img_rgb, gray):
        """Studio Ghibli landscape style"""
        # Soft, dreamy landscapes
        ghibli = cv2.bilateralFilter(img_rgb, 20, 100, 100)
        
        # Enhance natural colors
        hsv = cv2.cvtColor(ghibli, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.3)  # More vivid
        hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.1)  # Brighter
        ghibli = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add magical atmosphere
        h, w = gray.shape
        atmosphere = np.ones_like(img_rgb) * [245, 250, 255]  # Soft light
        
        # Gentle atmospheric blend
        for i in range(h):
            blend_factor = (i / h) * 0.1
            ghibli[i, :] = cv2.addWeighted(ghibli[i, :], 1-blend_factor,
                                          atmosphere[i, :], blend_factor, 0)
        
        # Soft edges
        edges = cv2.Canny(gray, 20, 60)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(ghibli, edges_colored // 6)
        return result
    
    def ghibli_character(self, img_rgb, gray):
        """Ghibli character style"""
        # Warm, approachable character rendering
        character = cv2.bilateralFilter(img_rgb, 25, 120, 120)
        
        # Warm color adjustment
        hsv = cv2.cvtColor(character, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = (hsv[:,:,0] - 10) % 180  # Warmer hues
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.2)
        character = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Gentle shading
        character = cv2.convertScaleAbs(character, alpha=1.1, beta=15)
        
        # Soft but defined edges
        edges = cv2.Canny(gray, 30, 80)
        edges = cv2.GaussianBlur(edges, (2, 2), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(character, edges_colored // 4)
        return result
    
    def spirited_away_style(self, img_rgb, gray):
        """Spirited Away magical style"""
        # Rich, magical atmosphere
        spirited = cv2.bilateralFilter(img_rgb, 20, 150, 150)
        
        # Enhance magical colors
        hsv = cv2.cvtColor(spirited, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.4)
        hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.2)
        spirited = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add magical particles
        h, w = gray.shape
        magic = np.zeros_like(img_rgb)
        for _ in range(150):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            if np.random.random() > 0.8:
                cv2.circle(magic, (x, y), np.random.randint(1, 3), (255, 255, 255), -1)
        
        # Mystical edges
        edges = cv2.Canny(gray, 25, 75)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.addWeighted(spirited, 0.9, magic, 0.1, 0)
        result = cv2.subtract(result, edges_colored // 5)
        
        return result
    
    def totoro_style(self, img_rgb, gray):
        """My Neighbor Totoro forest style"""
        # Lush forest greens
        totoro = cv2.bilateralFilter(img_rgb, 22, 130, 130)
        
        # Enhance greens and natural colors
        hsv = cv2.cvtColor(totoro, cv2.COLOR_RGB2HSV)
        # Boost green hues
        green_mask = (hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 85)
        hsv[:,:,1] = np.where(green_mask, np.minimum(hsv[:,:,1] * 1.5, 255), hsv[:,:,1])
        totoro = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add dappled sunlight effect
        h, w = gray.shape
        sunlight = np.zeros_like(img_rgb)
        for _ in range(20):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            radius = np.random.randint(20, 60)
            cv2.circle(sunlight, (x, y), radius, (30, 40, 10), -1)
        
        result = cv2.addWeighted(totoro, 0.9, sunlight, 0.1, 0)
        
        # Very soft edges
        edges = cv2.Canny(gray, 15, 50)
        edges = cv2.GaussianBlur(edges, (4, 4), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(result, edges_colored // 8)
        return result
    
    def mononoke_style(self, img_rgb, gray):
        """Princess Mononoke epic style"""
        # Epic, dramatic natural scenes
        mononoke = cv2.bilateralFilter(img_rgb, 18, 140, 140)
        
        # Enhance dramatic contrast
        mononoke = cv2.convertScaleAbs(mononoke, alpha=1.3, beta=-15)
        
        # Mystical color enhancement
        hsv = cv2.cvtColor(mononoke, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.3)
        mononoke = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add forest spirits effect
        h, w = gray.shape
        spirits = np.zeros_like(img_rgb)
        for _ in range(30):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            if np.random.random() > 0.9:
                cv2.circle(spirits, (x, y), 2, (200, 255, 200), -1)
        
        # Strong but organic edges
        edges = cv2.Canny(gray, 40, 100)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.addWeighted(mononoke, 0.95, spirits, 0.05, 0)
        result = cv2.subtract(result, edges_colored // 3)
        
        return result
    
    def howls_castle_style(self, img_rgb, gray):
        """Howl's Moving Castle steampunk magic"""
        # Steampunk magical atmosphere
        howl = cv2.bilateralFilter(img_rgb, 20, 120, 120)
        
        # Warm golden tones
        hsv = cv2.cvtColor(howl, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = (hsv[:,:,0] - 20) % 180  # Golden hues
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.2)
        howl = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add mechanical texture
        h, w = gray.shape
        mechanical = np.zeros_like(img_rgb)
        
        # Add gear-like patterns occasionally
        for _ in range(5):
            center = (np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4))
            cv2.circle(mechanical, center, 30, (20, 15, 10), 2)
            cv2.circle(mechanical, center, 20, (20, 15, 10), 1)
        
        # Magical sparkles
        sparkles = np.zeros_like(img_rgb)
        for _ in range(80):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            if np.random.random() > 0.85:
                cv2.circle(sparkles, (x, y), 1, (255, 220, 180), -1)
        
        result = cv2.addWeighted(howl, 0.85, mechanical, 0.05, 0)
        result = cv2.addWeighted(result, 0.95, sparkles, 0.05, 0)
        
        return result
    
    def castle_sky_style(self, img_rgb, gray):
        """Castle in the Sky adventure style"""
        # Adventure and wonder
        castle = cv2.bilateralFilter(img_rgb, 20, 110, 110)
        
        # Bright, adventurous colors
        castle = cv2.convertScaleAbs(castle, alpha=1.2, beta=25)
        
        # Sky-blue enhancement
        hsv = cv2.cvtColor(castle, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.3)
        castle = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add floating elements
        h, w = gray.shape
        floating = np.zeros_like(img_rgb)
        for _ in range(10):
            x, y = np.random.randint(0, w), np.random.randint(0, h//2)  # Upper half
            cv2.circle(floating, (x, y), np.random.randint(3, 8), (255, 255, 255), 1)
        
        result = cv2.addWeighted(castle, 0.95, floating, 0.05, 0)
        
        # Clean, adventure-style edges
        edges = cv2.Canny(gray, 35, 85)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(result, edges_colored // 4)
        return result
    
    def kikis_style(self, img_rgb, gray):
        """Kiki's Delivery Service whimsical style"""
        # Whimsical, European town atmosphere
        kiki = cv2.bilateralFilter(img_rgb, 22, 100, 100)
        
        # Warm, cozy colors
        hsv = cv2.cvtColor(kiki, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = (hsv[:,:,0] - 15) % 180  # Warmer
        hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.15)  # Brighter
        kiki = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add European charm texture
        h, w = gray.shape
        charm = np.random.randint(-8, 9, (h, w, 3), dtype=np.int16)
        kiki = np.clip(kiki.astype(np.int16) + charm, 0, 255).astype(np.uint8)
        
        # Gentle, storybook edges
        edges = cv2.Canny(gray, 25, 70)
        edges = cv2.GaussianBlur(edges, (2, 2), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(kiki, edges_colored // 5)
        return result
    
    def ponyo_style(self, img_rgb, gray):
        """Ponyo underwater magical style"""
        # Underwater magical world
        ponyo = cv2.bilateralFilter(img_rgb, 25, 140, 140)
        
        # Enhance blues and aquatic colors
        hsv = cv2.cvtColor(ponyo, cv2.COLOR_RGB2HSV)
        blue_mask = (hsv[:,:,0] >= 90) & (hsv[:,:,0] <= 140)
        hsv[:,:,1] = np.where(blue_mask, np.minimum(hsv[:,:,1] * 1.6, 255), hsv[:,:,1])
        ponyo = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add water bubbles
        h, w = gray.shape
        bubbles = np.zeros_like(img_rgb)
        for _ in range(100):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            if np.random.random() > 0.85:
                radius = np.random.randint(1, 5)
                cv2.circle(bubbles, (x, y), radius, (255, 255, 255), 1)
        
        # Very soft, flowing edges
        edges = cv2.Canny(gray, 20, 60)
        edges = cv2.GaussianBlur(edges, (4, 4), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.addWeighted(ponyo, 0.9, bubbles, 0.1, 0)
        result = cv2.subtract(result, edges_colored // 8)
        
        return result
    
    def wind_rises_style(self, img_rgb, gray):
        """The Wind Rises historical style"""
        # Historical, realistic but dreamy
        wind = cv2.bilateralFilter(img_rgb, 15, 100, 100)
        
        # Sepia-tinted historical look
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])
        sepia = wind.dot(sepia_filter.T)
        
        # Blend with original for subtle effect
        wind = cv2.addWeighted(wind, 0.7, sepia.astype(np.uint8), 0.3, 0)
        
        # Add period atmosphere
        wind = cv2.convertScaleAbs(wind, alpha=0.95, beta=10)
        
        # Realistic but soft edges
        edges = cv2.Canny(gray, 30, 80)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(wind, edges_colored // 4)
        return result
    
    def ghibli_sky(self, img_rgb, gray):
        """Ghibli dramatic sky style"""
        # Dramatic, ever-changing skies
        sky = cv2.bilateralFilter(img_rgb, 20, 120, 120)
        
        # Enhance sky colors
        hsv = cv2.cvtColor(sky, cv2.COLOR_RGB2HSV)
        # Enhance blues and warm colors
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.4)
        hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.2)
        sky = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add cloud texture
        h, w = gray.shape
        clouds = np.zeros_like(img_rgb)
        for _ in range(15):
            center = (np.random.randint(0, w), np.random.randint(0, h//2))
            axes = (np.random.randint(30, 80), np.random.randint(15, 40))
            cv2.ellipse(clouds, center, axes, 0, 0, 360, (20, 25, 30), -1)
        
        result = cv2.addWeighted(sky, 0.95, clouds, 0.05, 0)
        
        # Minimal edges for sky
        edges = cv2.Canny(gray, 20, 60)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(result, edges_colored // 8)
        return result
    
    def ghibli_forest(self, img_rgb, gray):
        """Ghibli enchanted forest"""
        # Deep, mystical forest
        forest = cv2.bilateralFilter(img_rgb, 25, 150, 150)
        
        # Rich forest greens
        hsv = cv2.cvtColor(forest, cv2.COLOR_RGB2HSV)
        green_mask = (hsv[:,:,0] >= 30) & (hsv[:,:,0] <= 90)
        hsv[:,:,1] = np.where(green_mask, np.minimum(hsv[:,:,1] * 1.6, 255), hsv[:,:,1])
        forest = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add mystical lighting
        h, w = gray.shape
        mystical = np.zeros_like(img_rgb)
        
        # Dappled light through trees
        for _ in range(25):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            radius = np.random.randint(10, 30)
            cv2.circle(mystical, (x, y), radius, (40, 50, 20), -1)
        
        # Forest spirits
        # Forest spirits
        spirits = np.zeros_like(img_rgb)
        for _ in range(50):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            if np.random.random() > 0.92:
                cv2.circle(spirits, (x, y), 1, (180, 255, 180), -1)
        
        # Combine all effects
        result = cv2.addWeighted(forest, 0.85, mystical, 0.1, 0)
        result = cv2.addWeighted(result, 0.95, spirits, 0.05, 0)
        
        # Soft, mystical edges
        edges = cv2.Canny(gray, 15, 45)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(result, edges_colored // 10)
        return result
    
    def ghibli_magic(self, img_rgb, gray):
        """Ghibli magical moments style"""
        # Magical, glowing atmosphere
        magic = cv2.bilateralFilter(img_rgb, 30, 160, 160)
        
        # Enhance magical colors
        hsv = cv2.cvtColor(magic, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.5)
        hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.3)
        magic = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add magical sparkles
        h, w = gray.shape
        sparkles = np.zeros_like(img_rgb)
        for _ in range(200):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            if np.random.random() > 0.88:
                size = np.random.randint(1, 3)
                color = (255, 255, 200) if np.random.random() > 0.5 else (200, 255, 255)
                cv2.circle(sparkles, (x, y), size, color, -1)
        
        # Add magical glow
        glow = cv2.GaussianBlur(sparkles, (15, 15), 0)
        
        result = cv2.addWeighted(magic, 0.8, sparkles, 0.1, 0)
        result = cv2.addWeighted(result, 0.9, glow, 0.1, 0)
        
        # Very soft edges for magical effect
        edges = cv2.Canny(gray, 10, 40)
        edges = cv2.GaussianBlur(edges, (7, 7), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(result, edges_colored // 12)
        return result
    
    def ghibli_nostalgia(self, img_rgb, gray):
        """Ghibli nostalgic, memory-like style"""
        # Warm, nostalgic atmosphere
        nostalgia = cv2.bilateralFilter(img_rgb, 18, 100, 100)
        
        # Warm, golden tones
        hsv = cv2.cvtColor(nostalgia, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = (hsv[:,:,0] - 20) % 180  # Warmer hues
        hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.1)  # Slightly brighter
        nostalgia = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add gentle golden overlay
        golden_overlay = np.full_like(img_rgb, [255, 240, 200])
        nostalgia = cv2.addWeighted(nostalgia, 0.85, golden_overlay, 0.15, 0)
        
        # Add memory-like softness
        soft = cv2.GaussianBlur(nostalgia, (3, 3), 0)
        result = cv2.addWeighted(nostalgia, 0.7, soft, 0.3, 0)
        
        # Gentle, dreamy edges
        edges = cv2.Canny(gray, 20, 60)
        edges = cv2.GaussianBlur(edges, (4, 4), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(result, edges_colored // 6)
        return result
    
    def ghibli_dreams(self, img_rgb, gray):
        """Ghibli dreamlike, surreal style"""
        # Dreamlike, ethereal atmosphere
        dreams = cv2.bilateralFilter(img_rgb, 35, 180, 180)
        
        # Enhance dream-like colors
        hsv = cv2.cvtColor(dreams, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.4)
        hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.2)
        dreams = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add dreamy clouds/mist
        h, w = gray.shape
        mist = np.zeros_like(img_rgb)
        for _ in range(8):
            center = (np.random.randint(0, w), np.random.randint(0, h))
            axes = (np.random.randint(50, 120), np.random.randint(30, 80))
            angle = np.random.randint(0, 180)
            cv2.ellipse(mist, center, axes, angle, 0, 360, (255, 255, 255), -1)
        
        mist = cv2.GaussianBlur(mist, (25, 25), 0)
        
        # Add floating dream elements
        dream_elements = np.zeros_like(img_rgb)
        for _ in range(30):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            if np.random.random() > 0.85:
                radius = np.random.randint(2, 6)
                alpha = np.random.randint(50, 150)
                cv2.circle(dream_elements, (x, y), radius, (255, 255, 255), -1)
        
        result = cv2.addWeighted(dreams, 0.8, mist, 0.1, 0)
        result = cv2.addWeighted(result, 0.95, dream_elements, 0.05, 0)
        
        # Very soft, ethereal edges
        edges = cv2.Canny(gray, 8, 35)
        edges = cv2.GaussianBlur(edges, (8, 8), 0)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        result = cv2.subtract(result, edges_colored // 15)
        return result

    def thick_impasto(self, img_rgb, gray):
        """Thick impasto acrylic technique with heavy texture"""
        # Create texture using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        textured = cv2.morphologyEx(img_rgb, cv2.MORPH_TOPHAT, kernel)
        
        # Blend with original for paint thickness effect
        impasto = cv2.addWeighted(img_rgb, 0.7, textured, 0.3, 0)
        
        # Enhance contrast for paint buildup
        enhanced = cv2.convertScaleAbs(impasto, alpha=1.3, beta=15)
        return enhanced

    def palette_knife(self, img_rgb, gray):
        """Palette knife acrylic painting technique"""
        # Create directional blur for knife strokes
        kernel = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)
        knife_strokes = cv2.filter2D(img_rgb, -1, kernel)
        
        # Add angular texture
        angular = cv2.Sobel(img_rgb, cv2.CV_8U, 1, 1, ksize=3)
        result = cv2.addWeighted(knife_strokes, 0.8, angular, 0.2, 0)
        
        # Boost saturation for acrylic vibrancy
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.4)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def wet_on_wet(self, img_rgb, gray):
        """Wet-on-wet acrylic blending technique"""
        # Multiple bilateral filters for smooth blending
        wet = cv2.bilateralFilter(img_rgb, 20, 100, 100)
        wet = cv2.bilateralFilter(wet, 15, 80, 80)
        
        # Add soft color bleeding effect
        kernel = np.ones((7,7), np.float32) / 49
        blended = cv2.filter2D(wet, -1, kernel)
        
        # Subtle color enhancement
        return cv2.convertScaleAbs(blended, alpha=1.1, beta=5)

    def dry_brush(self, img_rgb, gray):
        """Dry brush acrylic technique with textural strokes"""
        # Create brush texture using erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        dry = cv2.erode(img_rgb, kernel, iterations=1)
        
        # Add directional texture
        kernel_h = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.float32)
        horizontal = cv2.filter2D(dry, -1, kernel_h)
        
        # Blend for dry brush effect
        result = cv2.addWeighted(img_rgb, 0.6, horizontal, 0.4, 0)
        return cv2.convertScaleAbs(result, alpha=1.2, beta=10)

    def glazing_layers(self, img_rgb, gray):
        """Transparent glazing layers in acrylics"""
        # Create multiple transparent layers
        layer1 = cv2.convertScaleAbs(img_rgb, alpha=0.8, beta=20)
        layer2 = cv2.convertScaleAbs(img_rgb, alpha=0.9, beta=10)
        
        # Blend layers with transparency
        glazed = cv2.addWeighted(layer1, 0.5, layer2, 0.5, 0)
        
        # Add luminosity
        lab = cv2.cvtColor(glazed, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = cv2.add(lab[:,:,0], 15)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def acrylic_pour(self, img_rgb, gray):
        """Acrylic pouring fluid art effect"""
        # Create flowing patterns
        flow_x = cv2.Sobel(img_rgb, cv2.CV_32F, 1, 0, ksize=5)
        flow_y = cv2.Sobel(img_rgb, cv2.CV_32F, 0, 1, ksize=5)
        
        # Blend flows for pour effect
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Apply to all channels
        poured = np.zeros_like(img_rgb)
        for i in range(3):
            poured[:,:,i] = cv2.addWeighted(img_rgb[:,:,i], 0.7, normalized, 0.3, 0)
        
        return poured

    def scumbling(self, img_rgb, gray):
        """Scumbling technique with broken color application"""
        # Create broken texture pattern
        noise = np.random.random(img_rgb.shape[:2]) * 50
        noise = noise.astype(np.uint8)
        
        # Apply texture selectively
        mask = noise > 25
        scumbled = img_rgb.copy()
        
        for i in range(3):
            scumbled[:,:,i] = np.where(mask, 
                                    cv2.add(img_rgb[:,:,i], noise//2), 
                                    img_rgb[:,:,i])
        
        return cv2.convertScaleAbs(scumbled, alpha=1.1, beta=5)

    def color_blocking(self, img_rgb, gray):
        """Bold color blocking acrylic style"""
        # Reduce colors for blocking effect
        data = img_rgb.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
        _, labels, centers = cv2.kmeans(data, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and reshape
        centers = np.uint8(centers)
        blocked = centers[labels.flatten()]
        blocked = blocked.reshape(img_rgb.shape)
        
        return cv2.convertScaleAbs(blocked, alpha=1.2, beta=0)

    def textured_canvas(self, img_rgb, gray):
        """Canvas texture visible through acrylic paint"""
        # Create canvas weave pattern
        h, w = img_rgb.shape[:2]
        canvas = np.zeros((h, w), dtype=np.uint8)
        
        # Horizontal threads
        canvas[::4, :] = 30
        # Vertical threads  
        canvas[:, ::4] = 30
        
        # Apply canvas texture
        textured = img_rgb.copy()
        for i in range(3):
            textured[:,:,i] = cv2.subtract(textured[:,:,i], canvas//3)
        
        return cv2.convertScaleAbs(textured, alpha=1.1, beta=5)

    def heavy_body_acrylic(self, img_rgb, gray):
        """Heavy body acrylic with visible brush marks"""
        # Create brush mark texture
        kernel_v = np.array([[1], [2], [1]], dtype=np.float32)
        kernel_h = np.array([[1, 2, 1]], dtype=np.float32)
        
        vertical = cv2.filter2D(img_rgb, -1, kernel_v)
        horizontal = cv2.filter2D(img_rgb, -1, kernel_h)
        
        # Combine brush directions
        brushed = cv2.addWeighted(vertical, 0.5, horizontal, 0.5, 0)
        
        # Add paint thickness
        return cv2.convertScaleAbs(brushed, alpha=1.3, beta=20)

    def fluid_acrylic(self, img_rgb, gray):
        """Fluid acrylic smooth application"""
        # Smooth flowing effect
        smooth = cv2.GaussianBlur(img_rgb, (11, 11), 0)
        
        # Add subtle flow lines
        grad_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        
        # Normalize gradients
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        normalized = cv2.normalize(magnitude, None, 0, 50, cv2.NORM_MINMAX)
        
        # Apply flow effect
        result = cv2.addWeighted(smooth, 0.9, normalized.astype(np.uint8), 0.1, 0)
        return result

    def stippled_acrylic(self, img_rgb, gray):
        """Stippling technique with acrylic paint"""
        # Create stipple pattern
        stipple = img_rgb.copy()
        h, w = img_rgb.shape[:2]
        
        # Random stipple points
        for _ in range(h * w // 20):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            cv2.circle(stipple, (x, y), 1, (255, 255, 255), -1)
        
        # Blend with original
        return cv2.addWeighted(img_rgb, 0.8, stipple, 0.2, 0)

    def crosshatched_acrylic(self, img_rgb, gray):
        """Crosshatching with acrylic medium"""
        # Create crosshatch pattern
        kernel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
        kernel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        
        hatch1 = cv2.filter2D(gray, cv2.CV_32F, kernel1)
        hatch2 = cv2.filter2D(gray, cv2.CV_32F, kernel2)
        
        # Combine hatching
        crosshatch = cv2.addWeighted(np.abs(hatch1), 0.5, np.abs(hatch2), 0.5, 0)
        crosshatch = cv2.normalize(crosshatch, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Apply to color image
        result = img_rgb.copy()
        for i in range(3):
            result[:,:,i] = cv2.subtract(result[:,:,i], crosshatch//4)
        
        return result

    def acrylic_wash(self, img_rgb, gray):
        """Transparent acrylic wash technique"""
        # Create wash effect with reduced opacity
        washed = cv2.convertScaleAbs(img_rgb, alpha=0.7, beta=30)
        
        # Add water-like flow
        kernel = np.ones((9, 9), np.float32) / 81
        flowing = cv2.filter2D(washed, -1, kernel)
        
        # Blend for transparency
        return cv2.addWeighted(flowing, 0.8, img_rgb, 0.2, 0)

    def color_mixing(self, img_rgb, gray):
        """Visible color mixing on palette"""
        # Simulate color mixing by blending adjacent pixels
        mixed = cv2.medianBlur(img_rgb, 7)
        
        # Add mixing streaks
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        streaks = cv2.filter2D(mixed, cv2.CV_32F, kernel)
        streaks = cv2.convertScaleAbs(streaks)
        
        return cv2.addWeighted(mixed, 0.8, streaks, 0.2, 0)

    def bold_strokes(self, img_rgb, gray):
        """Bold, confident acrylic brush strokes"""
        # Enhance edges for bold strokes
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        # Apply bold enhancement
        bold = cv2.convertScaleAbs(img_rgb, alpha=1.4, beta=0)
        
        # Combine with edge definition
        for i in range(3):
            bold[:,:,i] = cv2.add(bold[:,:,i], edges//4)
        
        return bold

    def layered_acrylic(self, img_rgb, gray):
        """Multiple layered acrylic applications"""
        # Create multiple layers
        layer1 = cv2.GaussianBlur(img_rgb, (5, 5), 0)
        layer2 = cv2.bilateralFilter(img_rgb, 9, 75, 75)
        layer3 = cv2.medianBlur(img_rgb, 5)
        
        # Blend layers
        temp = cv2.addWeighted(layer1, 0.4, layer2, 0.4, 0)
        layered = cv2.addWeighted(temp, 0.7, layer3, 0.3, 0)
        
        return cv2.convertScaleAbs(layered, alpha=1.1, beta=5)

    def gestural_painting(self, img_rgb, gray):
        """Gestural acrylic painting style"""
        # Create gestural movements
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        gesture_x = cv2.filter2D(img_rgb, cv2.CV_32F, kernel)
        gesture_y = cv2.filter2D(img_rgb, cv2.CV_32F, kernel.T)
        
        # Combine gestures
        gestural = np.sqrt(gesture_x**2 + gesture_y**2)
        gestural = cv2.normalize(gestural, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Apply gestural effect
        result = cv2.addWeighted(img_rgb, 0.7, gestural, 0.3, 0)
        return result

    def acrylic_medium_gel(self, img_rgb, gray):
        """Acrylic gel medium texture effect"""
        # Create gel-like texture
        gel = cv2.morphologyEx(img_rgb, cv2.MORPH_TOPHAT, 
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        
        # Add transparency and gloss
        glossy = cv2.addWeighted(img_rgb, 0.8, gel, 0.2, 10)
        
        return cv2.convertScaleAbs(glossy, alpha=1.1, beta=0)

    def fan_brush_texture(self, img_rgb, gray):
        """Fan brush texture for acrylics"""
        # Create fan brush pattern
        h, w = img_rgb.shape[:2]
        fan_texture = np.zeros_like(img_rgb)
        
        # Simulate fan brush strokes
        for y in range(0, h, 8):
            for x in range(0, w, 12):
                # Create fan pattern
                cv2.ellipse(fan_texture, (x, y), (6, 2), 0, 0, 180, (255, 255, 255), 1)
        
        # Apply texture
        textured = cv2.addWeighted(img_rgb, 0.9, fan_texture, 0.1, 0)
        return textured

    def acrylic_sgraffito(self, img_rgb, gray):
        """Sgraffito technique - scratching through paint layers"""
        # Create scratched effect
        scratched = img_rgb.copy()
        
        # Simulate scratch marks
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8))
        
        # Apply scratches
        for i in range(3):
            scratched[:,:,i] = np.where(edges > 0, 
                                    cv2.add(scratched[:,:,i], 50), 
                                    scratched[:,:,i])
        
        return scratched

    def thick_paint_buildup(self, img_rgb, gray):
        """Thick paint buildup with visible texture"""
        # Create paint buildup effect
        buildup = cv2.morphologyEx(img_rgb, cv2.MORPH_CLOSE, 
                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        
        # Add texture variation
        kernel = np.random.random((5, 5)) - 0.5
        kernel = kernel.astype(np.float32)
        textured = cv2.filter2D(buildup, -1, kernel)
        
        return cv2.convertScaleAbs(textured, alpha=1.2, beta=15)

    def loose_brushwork(self, img_rgb, gray):
        """Loose, expressive brushwork"""
        # Create loose strokes
        loose = cv2.bilateralFilter(img_rgb, 15, 40, 40)
        
        # Add expressiveness
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        expressive = cv2.filter2D(loose, -1, kernel)
        
        return cv2.convertScaleAbs(expressive, alpha=1.1, beta=5)

    def acrylic_underpainting(self, img_rgb, gray):
        """Visible underpainting technique"""
        # Create underpainting base
        underpainting = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        underpainting = cv2.convertScaleAbs(underpainting, alpha=0.8, beta=20)
        
        # Blend with main painting
        result = cv2.addWeighted(img_rgb, 0.7, underpainting, 0.3, 0)
        
        return result

    def broken_color(self, img_rgb, gray):
        """Broken color technique with visible brushstrokes"""
        # Break up colors
        broken = img_rgb.copy()
        h, w = img_rgb.shape[:2]
        
        # Add color breaks
        for _ in range(h * w // 100):
            y, x = np.random.randint(2, h-2), np.random.randint(2, w-2)
            # Small color variations
            broken[y-1:y+2, x-1:x+2] = cv2.add(broken[y-1:y+2, x-1:x+2], 
                                            np.random.randint(-20, 20, (3, 3, 3)))
        
        return np.clip(broken, 0, 255)

    def acrylic_glazing(self, img_rgb, gray):
        """Traditional glazing with acrylic medium"""
        # Multiple transparent glazes
        glaze1 = cv2.convertScaleAbs(img_rgb, alpha=0.9, beta=5)
        glaze2 = cv2.convertScaleAbs(img_rgb, alpha=0.8, beta=10)
        glaze3 = cv2.convertScaleAbs(img_rgb, alpha=0.85, beta=7)
        
        # Layer glazes
        temp = cv2.addWeighted(glaze1, 0.4, glaze2, 0.3, 0)
        glazed = cv2.addWeighted(temp, 0.7, glaze3, 0.3, 0)
        
        return glazed

    def painterly_edges(self, img_rgb, gray):
        """Soft painterly edges"""
        # Soften edges
        soft = cv2.bilateralFilter(img_rgb, 20, 80, 80)
        
        # Maintain some edge definition
        edges = cv2.Canny(gray, 30, 90)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        
        # Combine
        for i in range(3):
            soft[:,:,i] = cv2.add(soft[:,:,i], edges//8)
        
        return soft

    def acrylic_pouring_cells(self, img_rgb, gray):
        """Acrylic pouring with cell formation"""
        # Create cell-like patterns
        cells = cv2.morphologyEx(img_rgb, cv2.MORPH_OPEN, 
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        
        # Add cell boundaries
        diff = cv2.absdiff(img_rgb, cells)
        cells_enhanced = cv2.add(cells, diff//2)
        
        return cv2.convertScaleAbs(cells_enhanced, alpha=1.1, beta=5)

    def high_key_acrylic(self, img_rgb, gray):
        """High key acrylic painting style"""
        # Brighten overall
        high_key = cv2.convertScaleAbs(img_rgb, alpha=1.1, beta=30)
        
        # Reduce contrast slightly
        high_key = cv2.addWeighted(high_key, 0.8, 
                                np.full_like(high_key, 128), 0.2, 0)
        
        return high_key

    def low_key_acrylic(self, img_rgb, gray):
        """Low key dramatic acrylic style"""
        # Darken and increase contrast
        low_key = cv2.convertScaleAbs(img_rgb, alpha=1.3, beta=-30)
        
        # Enhance shadows
        shadows = np.where(gray < 100, gray, 100)
        for i in range(3):
            low_key[:,:,i] = cv2.multiply(low_key[:,:,i], shadows/255.0)
        
        return low_key

    def expressive_color(self, img_rgb, gray):
        """Expressive non-naturalistic color"""
        # Shift colors expressively  
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        # Shift hues
        hsv[:,:,0] = cv2.add(hsv[:,:,0], 30)
        # Boost saturation
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.5)
        
        expressive = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return np.clip(expressive, 0, 255)

    def acrylic_collage(self, img_rgb, gray):
        """Mixed media acrylic collage effect"""
        # Create collage-like texture
        collage = img_rgb.copy()
        
        # Add paper-like texture
        h, w = img_rgb.shape[:2]
        paper_texture = np.random.normal(0, 10, (h, w)).astype(np.int16)
        
        for i in range(3):
            collage[:,:,i] = np.clip(collage[:,:,i] + paper_texture, 0, 255)
        
        return collage.astype(np.uint8)

    def rhythmic_brushwork(self, img_rgb, gray):
        """Rhythmic, musical brushwork pattern"""
        # Create rhythmic pattern
        rhythmic = img_rgb.copy()
        h, w = img_rgb.shape[:2]
        
        # Add rhythmic strokes
        for y in range(0, h, 15):
            rhythm_intensity = int(20 * np.sin(y * 0.1))
            rhythmic[y:y+3, :] = cv2.add(rhythmic[y:y+3, :], rhythm_intensity)
        
        return np.clip(rhythmic, 0, 255)

    def acrylic_transparency(self, img_rgb, gray):
        """Play with acrylic transparency effects"""
        # Create transparency variations
        transparent = img_rgb.copy().astype(np.float32)
        
        # Vary transparency based on brightness
        brightness = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        alpha_mask = brightness / 255.0
        
        # Apply transparency effect
        for i in range(3):
            transparent[:,:,i] = transparent[:,:,i] * alpha_mask + 255 * (1 - alpha_mask) * 0.1
        
        return np.clip(transparent, 0, 255).astype(np.uint8)

    def directional_strokes(self, img_rgb, gray):
        """Strong directional brush strokes"""
        # Create directional blur
        kernel = np.zeros((7, 7), np.float32)
        kernel[3, :] = 1/7  # Horizontal strokes
        
        directional = cv2.filter2D(img_rgb, -1, kernel)
        
        # Blend with original
        return cv2.addWeighted(img_rgb, 0.6, directional, 0.4, 0)

    def acrylic_spray(self, img_rgb, gray):
        """Spray paint acrylic technique"""
        # Create spray pattern
        spray = img_rgb.copy()
        h, w = img_rgb.shape[:2]
        
        # Add spray dots
        for _ in range(h * w // 50):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            intensity = np.random.randint(200, 255)
            cv2.circle(spray, (x, y), 1, (intensity, intensity, intensity), -1)
        
        # Blend spray effect
        return cv2.addWeighted(img_rgb, 0.8, spray, 0.2, 0)

    def color_temperature_mix(self, img_rgb, gray):
        """Warm and cool color temperature mixing"""
        # Split into warm and cool areas
        warm = img_rgb.copy().astype(np.float32)
        cool = img_rgb.copy().astype(np.float32)
        
        # Enhance warm tones (reds, oranges, yellows)
        warm[:,:,0] = np.minimum(warm[:,:,0] * 1.2, 255)  # Red
        warm[:,:,1] = np.minimum(warm[:,:,1] * 1.1, 255)  # Green
        
        # Enhance cool tones (blues, purples)
        cool[:,:,2] = np.minimum(cool[:,:,2] * 1.2, 255)  # Blue
        cool[:,:,1] = cool[:,:,1] * 0.9  # Reduce green for cooler tone
        
        # Mix based on original color temperature
        temp_mask = (img_rgb[:,:,0] + img_rgb[:,:,1]) > img_rgb[:,:,2]
        temp_mask = temp_mask.astype(np.float32)[:,:,np.newaxis]
        
        mixed = warm * temp_mask + cool * (1 - temp_mask)
        return np.clip(mixed, 0, 255).astype(np.uint8)

    def acrylic_sketch_underlay(self, img_rgb, gray):
        """Visible sketch under acrylic paint"""
        # Create sketch layer
        edges = cv2.Canny(gray, 50, 100)
        sketch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Thin the sketch lines
        sketch = cv2.erode(sketch, np.ones((2,2), np.uint8))
        
        # Overlay sketch under paint
        result = cv2.addWeighted(img_rgb, 0.85, sketch, 0.15, 0)
        return result

    def loose_color_application(self, img_rgb, gray):
        """Loose, free color application"""
        # Create loose application effect
        loose = cv2.medianBlur(img_rgb, 9)
        
        # Add color variations
        h, w = img_rgb.shape[:2]
        variations = np.random.randint(-15, 15, img_rgb.shape).astype(np.int16)
        
        loose_varied = loose.astype(np.int16) + variations
        return np.clip(loose_varied, 0, 255).astype(np.uint8)

    def acrylic_texture_paste(self, img_rgb, gray):
        """Texture paste mixed with acrylics"""
        # Create texture paste effect
        texture = cv2.morphologyEx(img_rgb, cv2.MORPH_TOPHAT, 
                                cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        
        # Add granular texture
        granular = cv2.addWeighted(img_rgb, 0.8, texture, 0.2, 0)
        
        # Enhance texture visibility
        return cv2.convertScaleAbs(granular, alpha=1.15, beta=5)

    def complementary_color_play(self, img_rgb, gray):
        """Playing with complementary color relationships"""
        # Convert to HSV for hue manipulation
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        # Create complementary version
        comp_hsv = hsv.copy()
        comp_hsv[:,:,0] = cv2.add(comp_hsv[:,:,0], 90)  # Shift hue by ~180°
        
        complementary = cv2.cvtColor(comp_hsv, cv2.COLOR_HSV2RGB)
        
        # Blend strategically
        mask = gray > 128
        result = img_rgb.copy()
        result[mask] = cv2.addWeighted(img_rgb[mask], 0.7, complementary[mask], 0.3, 0)
        
        return result

    def energetic_brushwork(self, img_rgb, gray):
        """High energy, dynamic brushwork"""
        # Create energy through motion blur variations
        energy = img_rgb.copy()
        
        # Multiple directional blurs
        kernel1 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32) / 3
        kernel2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32) / 3
        
        energy1 = cv2.filter2D(energy, -1, kernel1)
        energy2 = cv2.filter2D(energy, -1, kernel2)
        
        # Combine energetic strokes
        energetic = cv2.addWeighted(energy1, 0.5, energy2, 0.5, 0)
        return cv2.convertScaleAbs(energetic, alpha=1.2, beta=10)

    def acrylic_color_studies(self, img_rgb, gray):
        """Color study approach with limited palette"""
        # Reduce to study palette
        data = img_rgb.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back and enhance
        centers = np.uint8(centers)
        study = centers[labels.flatten()]
        study = study.reshape(img_rgb.shape)
        
        return cv2.convertScaleAbs(study, alpha=1.1, beta=5)



    def monochromatic_acrylic(self, img_rgb, gray):
        """Monochromatic acrylic painting in single hue"""
        # Convert to single hue variations
        mono = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Add hue tint (blue tint example)
        mono[:,:,2] = cv2.add(mono[:,:,2], 30)  # Blue channel
        
        # Vary saturation based on original colors
        hsv_orig = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        saturation_map = hsv_orig[:,:,1]
        
        # Apply saturation variation
        mono = cv2.addWeighted(mono, 0.8, cv2.cvtColor(saturation_map, cv2.COLOR_GRAY2RGB), 0.2, 0)
        
        return mono

    def acrylic_washes_layered(self, img_rgb, gray):
        """Multiple transparent acrylic washes"""
        # Create multiple wash layers
        wash1 = cv2.convertScaleAbs(img_rgb, alpha=0.6, beta=40)
        wash2 = cv2.convertScaleAbs(img_rgb, alpha=0.7, beta=20)
        wash3 = cv2.convertScaleAbs(img_rgb, alpha=0.8, beta=10)
        
        # Layer the washes
        combined = cv2.addWeighted(wash1, 0.3, wash2, 0.3, 0)
        layered_washes = cv2.addWeighted(combined, 0.6, wash3, 0.4, 0)
        
        return layered_washes

    def spontaneous_application(self, img_rgb, gray):
        """Spontaneous, intuitive paint application"""
        # Create spontaneous texture
        spontaneous = img_rgb.copy()
        h, w = img_rgb.shape[:2]
        
        # Random paint dabs
        for _ in range(h * w // 200):
            y = np.random.randint(5, h-5)
            x = np.random.randint(5, w-5)
            size = np.random.randint(2, 6)
            color_shift = np.random.randint(-30, 30, 3)
            
            # Apply random color dab
            roi = spontaneous[y-size:y+size, x-size:x+size]
            roi_shifted = np.clip(roi.astype(np.int16) + color_shift, 0, 255)
            spontaneous[y-size:y+size, x-size:x+size] = roi_shifted.astype(np.uint8)
        
        return spontaneous

    def acrylic_resist_technique(self, img_rgb, gray):
        """Resist technique with acrylic medium"""
        # Create resist pattern
        resist_mask = gray > 180  # Bright areas resist paint
        
        # Apply resist effect
        resisted = img_rgb.copy()
        
        # Darken non-resist areas
        resisted[~resist_mask] = cv2.convertScaleAbs(resisted[~resist_mask], alpha=0.7, beta=-20)
        
        # Enhance resist areas
        resisted[resist_mask] = cv2.convertScaleAbs(resisted[resist_mask], alpha=1.1, beta=20)
        
        return resisted

    def bold_color_contrasts(self, img_rgb, gray):
        """Bold, high-contrast color relationships"""
        # Enhance contrast dramatically
        contrasted = cv2.convertScaleAbs(img_rgb, alpha=1.5, beta=-50)
        
        # Push colors to extremes
        hsv = cv2.cvtColor(contrasted, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.8)  # Super saturate
        hsv[:,:,2] = cv2.convertScaleAbs(hsv[:,:,2], alpha=1.3, beta=0)  # Increase value
        
        bold = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return np.clip(bold, 0, 255)

    def acrylic_dry_layering(self, img_rgb, gray):
        """Dry layering technique with visible brush marks"""
        # Create dry layer base
        dry_base = cv2.convertScaleAbs(img_rgb, alpha=0.9, beta=0)
        
        # Add dry brush texture
        kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        dry_strokes = cv2.filter2D(dry_base, cv2.CV_32F, kernel)
        dry_strokes = cv2.convertScaleAbs(dry_strokes)
        
        # Layer multiple dry applications
        layer1 = cv2.addWeighted(dry_base, 0.7, dry_strokes, 0.3, 0)
        
        # Second dry layer
        kernel2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        dry_strokes2 = cv2.filter2D(layer1, cv2.CV_32F, kernel2)
        dry_strokes2 = cv2.convertScaleAbs(dry_strokes2)
        
        final_dry = cv2.addWeighted(layer1, 0.8, dry_strokes2, 0.2, 0)
        return final_dry

    def contemporary_acrylic(self, img_rgb, gray):
        """Contemporary acrylic painting style"""
        # Modern, clean acrylic application
        contemporary = cv2.bilateralFilter(img_rgb, 12, 60, 60)
        
        # Add contemporary color treatment
        lab = cv2.cvtColor(contemporary, cv2.COLOR_RGB2LAB)
        lab[:,:,1] = cv2.convertScaleAbs(lab[:,:,1], alpha=1.2, beta=0)  # Enhance a* channel
        lab[:,:,2] = cv2.convertScaleAbs(lab[:,:,2], alpha=1.2, beta=0)  # Enhance b* channel
        
        contemporary = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Clean, precise edges
        return cv2.convertScaleAbs(contemporary, alpha=1.1, beta=5)
    # PHOTOSHOP EFFECTS (131-140)

    def gaussian_blur_effect(self, img_rgb, gray):
        """Classic Photoshop Gaussian Blur"""
        # Apply strong Gaussian blur
        blurred = cv2.GaussianBlur(img_rgb, (21, 21), 8.0)
        
        # Optional blend with original for partial blur
        result = cv2.addWeighted(img_rgb, 0.3, blurred, 0.7, 0)
        return result

    def motion_blur_effect(self, img_rgb, gray):
        """Motion Blur filter effect"""
        # Create motion blur kernel
        kernel_size = 15
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Apply motion blur
        motion_blurred = cv2.filter2D(img_rgb, -1, kernel)
        return motion_blurred

    def emboss_filter(self, img_rgb, gray):
        """Emboss filter effect"""
        # Emboss kernel
        kernel = np.array([[-2, -1, 0],
                        [-1,  1, 1],
                        [ 0,  1, 2]], dtype=np.float32)
        
        # Apply emboss
        embossed = cv2.filter2D(img_rgb, cv2.CV_32F, kernel)
        embossed = cv2.convertScaleAbs(embossed)
        
        # Add neutral gray base for better effect
        embossed = cv2.add(embossed, 128)
        return np.clip(embossed, 0, 255)

    def edge_enhance(self, img_rgb, gray):
        """Edge Enhancement filter"""
        # Enhanced edge detection
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8))
        
        # Apply to all channels
        enhanced = img_rgb.copy()
        for i in range(3):
            enhanced[:,:,i] = cv2.add(enhanced[:,:,i], edges//2)
        
        # Sharpen the result
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return np.clip(sharpened, 0, 255)

    def find_edges_filter(self, img_rgb, gray):
        """Find Edges filter (like Photoshop)"""
        # Sobel edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine edges
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Invert for white background
        edges = 255 - edges
        
        # Convert to RGB
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    def radial_blur(self, img_rgb, gray):
        """Radial Blur effect"""
        h, w = img_rgb.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Create radial blur
        result = np.zeros_like(img_rgb, dtype=np.float32)
        
        for angle in range(0, 360, 10):  # Sample angles
            # Rotate image
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            rotated = cv2.warpAffine(img_rgb.astype(np.float32), M, (w, h))
            result += rotated
        
        # Average all rotations
        result = result / 36  # 360/10 = 36 samples
        return np.clip(result, 0, 255).astype(np.uint8)

    def lens_flare_effect(self, img_rgb, gray):
        """Lens Flare effect"""
        h, w = img_rgb.shape[:2]
        flare = img_rgb.copy().astype(np.float32)
        
        # Create lens flare spots
        flare_x, flare_y = w // 3, h // 4
        
        # Main flare
        cv2.circle(flare, (flare_x, flare_y), 80, (255, 255, 200), -1)
        
        # Secondary flares
        cv2.circle(flare, (flare_x + 100, flare_y + 50), 30, (255, 200, 150), -1)
        cv2.circle(flare, (flare_x + 150, flare_y + 80), 15, (255, 150, 100), -1)
        
        # Blur the flares
        flare = cv2.GaussianBlur(flare, (51, 51), 20)
        
        # Blend with original using screen blend mode
        result = 255 - (255 - img_rgb.astype(np.float32)) * (255 - flare) / 255
        return np.clip(result, 0, 255).astype(np.uint8)

    def plastic_wrap_effect(self, img_rgb, gray):
        """Plastic Wrap artistic filter"""
        # High-pass filter effect
        blurred = cv2.GaussianBlur(img_rgb, (21, 21), 10)
        high_pass = cv2.subtract(img_rgb, blurred)
        high_pass = cv2.add(high_pass, 128)
        
        # Apply plastic-like distortion
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        plastic = cv2.filter2D(high_pass, -1, kernel)
        
        # Enhance the effect
        return cv2.convertScaleAbs(plastic, alpha=1.5, beta=0)

    def chrome_effect(self, img_rgb, gray):
        """Chrome/Metallic effect"""
        # Convert to grayscale for metallic base
        chrome = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Enhance contrast for metallic look
        chrome = cv2.convertScaleAbs(chrome, alpha=2.0, beta=-100)
        
        # Add metallic highlights
        highlights = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        highlights = cv2.GaussianBlur(highlights, (5, 5), 0)
        
        # Apply highlights to all channels
        for i in range(3):
            chrome[:,:,i] = cv2.add(chrome[:,:,i], highlights//3)
        
        # Add slight blue tint for chrome effect
        chrome[:,:,2] = cv2.add(chrome[:,:,2], 20)
        
        return np.clip(chrome, 0, 255)

    def glowing_edges(self, img_rgb, gray):
        """Glowing Edges effect"""
        # Find edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Create glow effect
        glow = cv2.GaussianBlur(edges, (15, 15), 5)
        
        # Create colored glow
        glowing = np.zeros_like(img_rgb)
        glowing[:,:,0] = glow  # Red glow
        glowing[:,:,1] = glow // 2  # Some green
        glowing[:,:,2] = glow  # Blue glow
        
        # Blend with dark background
        dark_bg = cv2.convertScaleAbs(img_rgb, alpha=0.3, beta=0)
        result = cv2.add(dark_bg, glowing)
        
        return result

    # ILLUSTRATION EFFECTS (141-150)

    def vector_illustration(self, img_rgb, gray):
        """Vector-style illustration"""
        # Reduce colors for vector look
        data = img_rgb.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back
        centers = np.uint8(centers)
        vector = centers[labels.flatten()]
        vector = vector.reshape(img_rgb.shape)
        
        # Add clean edges
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8))
        
        # Apply edges to darken boundaries
        for i in range(3):
            vector[:,:,i] = cv2.subtract(vector[:,:,i], edges//2)
        
        return vector

    def flat_design_style(self, img_rgb, gray):
        """Flat design illustration style"""
        # Reduce to flat colors
        flat = cv2.medianBlur(img_rgb, 15)
        
        # Reduce color depth
        flat = (flat // 32) * 32  # Quantize colors
        
        # Enhance saturation for flat design pop
        hsv = cv2.cvtColor(flat, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.4)
        flat = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return np.clip(flat, 0, 255)

    def line_art_illustration(self, img_rgb, gray):
        """Clean line art style"""
        # Strong edge detection
        edges = cv2.Canny(gray, 80, 160)
        
        # Thin the lines
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Create clean line art on white background
        line_art = np.full_like(img_rgb, 255)  # White background
        
        # Add black lines
        for i in range(3):
            line_art[:,:,i] = cv2.subtract(line_art[:,:,i], edges)
        
        return line_art

    def watercolor_illustration(self, img_rgb, gray):
        """Watercolor illustration style"""
        # Create watercolor base
        watercolor = cv2.bilateralFilter(img_rgb, 15, 80, 80)
        watercolor = cv2.bilateralFilter(watercolor, 15, 80, 80)
        
        # Add watercolor bleeding effect
        kernel = np.ones((7, 7), np.float32) / 49
        bleeding = cv2.filter2D(watercolor, -1, kernel)
        
        # Blend for transparency effect
        result = cv2.addWeighted(watercolor, 0.7, bleeding, 0.3, 0)
        
        # Reduce intensity for watercolor transparency
        return cv2.convertScaleAbs(result, alpha=0.8, beta=20)

    def pen_ink_illustration(self, img_rgb, gray):
        """Pen and ink illustration style"""
        # Create crosshatch pattern based on darkness
        pen_ink = np.full_like(img_rgb, 255)  # White background
        
        # Create different hatch densities based on gray levels
        dark_areas = gray < 100
        medium_areas = (gray >= 100) & (gray < 180)
        
        # Dense hatching for dark areas
        hatch_dense = np.zeros_like(gray)
        hatch_dense[::2, :] = 255  # Horizontal lines
        hatch_dense[:, ::2] = 255  # Vertical lines
        
        # Medium hatching
        hatch_medium = np.zeros_like(gray)
        hatch_medium[::4, :] = 255
        
        # Apply hatching
        for i in range(3):
            pen_ink[:,:,i] = np.where(dark_areas, 255 - hatch_dense, pen_ink[:,:,i])
            pen_ink[:,:,i] = np.where(medium_areas, 255 - hatch_medium//2, pen_ink[:,:,i])
        
        return pen_ink

    def digital_painting_style(self, img_rgb, gray):
        """Digital painting illustration"""
        # Create painterly base
        digital = cv2.bilateralFilter(img_rgb, 12, 50, 50)
        
        # Add digital brush strokes
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        strokes = cv2.filter2D(digital, cv2.CV_32F, kernel)
        strokes = cv2.convertScaleAbs(strokes)
        
        # Blend strokes
        painted = cv2.addWeighted(digital, 0.8, strokes, 0.2, 0)
        
        # Enhance colors for digital vibrancy
        return cv2.convertScaleAbs(painted, alpha=1.2, beta=10)

    def cartoon_illustration(self, img_rgb, gray):
        """Cartoon/animation style"""
        # Strong bilateral filter for cartoon smoothness
        cartoon = cv2.bilateralFilter(img_rgb, 15, 100, 100)
        cartoon = cv2.bilateralFilter(cartoon, 15, 100, 100)
        
        # Reduce colors
        cartoon = (cartoon // 25) * 25
        
        # Add cartoon outlines
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8))
        
        # Apply thick black outlines
        for i in range(3):
            cartoon[:,:,i] = cv2.subtract(cartoon[:,:,i], edges)
        
        return cartoon

    def sketch_illustration(self, img_rgb, gray):
        """Pencil sketch illustration"""
        # Invert grayscale
        inverted = 255 - gray
        
        # Blur inverted image
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        
        # Create sketch by dividing
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        
        # Convert to RGB
        sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        
        # Add slight color tint from original
        tinted = cv2.addWeighted(sketch_rgb, 0.8, img_rgb, 0.2, 0)
        
        return tinted

    def minimalist_illustration(self, img_rgb, gray):
        """Minimalist illustration style"""
        # Extreme color reduction
        data = img_rgb.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        _, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back
        centers = np.uint8(centers)
        minimal = centers[labels.flatten()]
        minimal = minimal.reshape(img_rgb.shape)
        
        # High contrast for minimalist look
        return cv2.convertScaleAbs(minimal, alpha=1.3, beta=0)

    def isometric_illustration(self, img_rgb, gray):
        """Isometric illustration style"""
        # Create geometric/isometric feel
        geometric = cv2.bilateralFilter(img_rgb, 9, 50, 50)
        
        # Add geometric structure
        h, w = img_rgb.shape[:2]
        
        # Create diamond/isometric grid overlay
        overlay = np.zeros_like(geometric)
        
        # Add subtle geometric patterns
        for y in range(0, h, 20):
            for x in range(0, w, 20):
                # Small geometric shapes
                cv2.rectangle(overlay, (x, y), (x+15, y+15), (20, 20, 20), 1)
        
        # Blend geometric overlay
        isometric = cv2.addWeighted(geometric, 0.95, overlay, 0.05, 0)
        
        # Enhance for clean geometric look
        return cv2.convertScaleAbs(isometric, alpha=1.1, beta=5)
    def hologram_interference(self, img_rgb, gray):
        """Holographic interference pattern"""
        h, w = img_rgb.shape[:2]
        result = img_rgb.copy()
        
        # Create interference pattern
        x, y = np.meshgrid(np.linspace(0, 10, w), np.linspace(0, 10, h))
        
        # Multiple wave interference
        wave1 = np.sin(x * 2 * np.pi + y * 1.5 * np.pi)
        wave2 = np.sin(x * 1.8 * np.pi - y * 2.2 * np.pi)
        wave3 = np.sin((x + y) * 1.2 * np.pi)
        interference = (wave1 + wave2 + wave3) / 3
        
        # Convert to color modulation
        rainbow_phase = (interference + 1) * np.pi
        
        # Create rainbow colors
        rainbow = np.zeros((h, w, 3))
        rainbow[:, :, 0] = (np.sin(rainbow_phase) + 1) * 0.5  # Red channel
        rainbow[:, :, 1] = (np.sin(rainbow_phase + 2 * np.pi / 3) + 1) * 0.5  # Green channel
        rainbow[:, :, 2] = (np.sin(rainbow_phase + 4 * np.pi / 3) + 1) * 0.5  # Blue channel
        
        # Apply holographic effect
        # Use grayscale as base intensity
        gray_norm = gray.astype(np.float32) / 255.0
        gray_3d = np.stack([gray_norm, gray_norm, gray_norm], axis=2)
        
        # Modulate the original image with interference pattern
        interference_strength = 0.6  # Adjust for desired effect intensity
        interference_3d = np.stack([interference, interference, interference], axis=2)
        
        # Combine original image with holographic interference
        result = result.astype(np.float32) / 255.0
        holographic = result * (1 + interference_3d * interference_strength)
        
        # Add rainbow shimmer effect
        shimmer_strength = 0.3
        holographic = holographic * (1 - shimmer_strength) + rainbow * shimmer_strength * gray_3d
        
        # Enhance bright areas with more holographic effect
        brightness = np.mean(result, axis=2, keepdims=True)
        bright_mask = (brightness > 0.7).astype(np.float32)
        holographic = holographic + rainbow * bright_mask * 0.2
        
        # Normalize and convert back to uint8
        holographic = np.clip(holographic, 0, 1)
        result = (holographic * 255).astype(np.uint8)
        
    
    # MODERN/CYBERPUNK STYLES (151-160)
    
    def synthwave_style(self, img_rgb, gray):
        """80s Synthwave aesthetic with neon colors and grid lines"""
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.8, 0, 255)  # Boost saturation
        
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add neon glow effect
        glow = cv2.GaussianBlur(result, (21, 21), 0)
        result = cv2.addWeighted(result, 0.7, glow, 0.3, 0)
        
        # Add horizontal scan lines
        h, w = result.shape[:2]
        for i in range(0, h, 3):
            if i < h:
                result[i] = cv2.addWeighted(result[i], 0.8, 
                                          np.full_like(result[i], [255, 0, 255]), 0.2, 0)
        
        return cv2.convertScaleAbs(result, alpha=1.3, beta=10)
    
    def vaporwave_style(self, img_rgb, gray):
        """Vaporwave aesthetic with pastel colors and dreamy effect"""
        dreamy = cv2.GaussianBlur(img_rgb, (25, 25), 0)
        
        # Shift to pastel colors
        lab = cv2.cvtColor(dreamy, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = cv2.add(lab[:,:,0], 20)  # Brighten
        lab[:,:,1] = cv2.subtract(lab[:,:,1], 15)  # Reduce saturation
        lab[:,:,2] = cv2.add(lab[:,:,2], 25)  # Add blue tint
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Add gradient overlay
        h, w = result.shape[:2]
        gradient = np.linspace(0, 1, h).reshape(-1, 1, 1)
        gradient = np.repeat(gradient, w, axis=1)
        gradient = np.repeat(gradient, 3, axis=2)
        
        overlay = np.array([255, 192, 203]) * gradient
        return cv2.addWeighted(result, 0.75, overlay.astype(np.uint8), 0.25, 0)
    
    def glitch_art_style(self, img_rgb, gray):
        """Digital glitch effect with RGB channel separation"""
        h, w, c = img_rgb.shape
        result = img_rgb.copy()
        
        # RGB channel shift
        r, g, b = cv2.split(result)
        
        shift_r = np.zeros_like(r)
        shift_g = np.zeros_like(g)
        
        offset = 8
        shift_r[:, offset:] = r[:, :-offset]
        shift_g[:, :-offset] = g[:, offset:]
        
        glitched = cv2.merge([shift_r, shift_g, b])
        
        # Add glitch lines
        for _ in range(15):
            y = np.random.randint(0, h-2)
            thickness = np.random.randint(1, 4)
            color = [np.random.randint(0, 255) for _ in range(3)]
            glitched[y:y+thickness, :] = color
        
        return glitched
    
    def holographic_effect(self, img_rgb, gray):
        """Iridescent holographic rainbow effect"""
        h, w = img_rgb.shape[:2]
        
        # Create iridescent effect
        x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        pattern = np.sin(x * 10 + y * 10) * 0.5 + 0.5
        
        # Rainbow color mapping
        rainbow = np.zeros((h, w, 3), dtype=np.uint8)
        hue = (pattern * 360).astype(np.uint8)
        sat = np.full((h, w), 255, dtype=np.uint8)
        val = np.full((h, w), 255, dtype=np.uint8)
        
        hsv = cv2.merge([hue, sat, val])
        rainbow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Blend with original
        result = cv2.addWeighted(img_rgb, 0.6, rainbow, 0.4, 0)
        
        # Add interference lines
        for i in range(0, h, 8):
            if i < h-1:
                result[i:i+1] = cv2.addWeighted(result[i:i+1], 0.7, 
                                              np.full_like(result[i:i+1], 255), 0.3, 0)
        
        return result
    
    def matrix_digital(self, img_rgb, gray):
        """Matrix-style digital rain effect"""
        # Green tint
        matrix = img_rgb.copy()
        matrix[:,:,0] = matrix[:,:,0] * 0.2  # Reduce red
        matrix[:,:,2] = matrix[:,:,2] * 0.2  # Reduce blue
        matrix[:,:,1] = np.clip(matrix[:,:,1] * 1.5, 0, 255)  # Boost green
        
        # Add digital noise
        h, w = matrix.shape[:2]
        noise = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
        matrix = cv2.add(matrix, noise)
        
        # Add vertical streaks
        for _ in range(20):
            x = np.random.randint(0, w)
            length = np.random.randint(h//4, h//2)
            y_start = np.random.randint(0, h-length)
            
            for i in range(length):
                fade = 1 - (i / length)
                intensity = int(255 * fade)
                if y_start + i < h:
                    matrix[y_start + i, x] = [0, intensity, 0]
        
        return matrix
    
    def tron_legacy_style(self, img_rgb, gray):
        """Tron Legacy neon outline style"""
        # Edge detection
        edges = cv2.Canny(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), 50, 150)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8))
        
        # Create dark base
        dark = img_rgb * 0.2
        
        # Neon edges
        neon_edges = np.zeros_like(img_rgb)
        neon_edges[edges > 0] = [0, 255, 255]  # Cyan neon
        
        # Glow effect
        glow = cv2.GaussianBlur(neon_edges, (15, 15), 0)
        
        result = cv2.add(dark.astype(np.uint8), neon_edges)
        result = cv2.addWeighted(result, 0.8, glow, 0.4, 0)
        
        return result
    
    def blade_runner_aesthetic(self, img_rgb, gray):
        """Blade Runner cyberpunk atmosphere"""
        # Orange-blue color grading
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        lab[:,:,1] = cv2.add(lab[:,:,1], 20)  # Orange tint
        lab[:,:,2] = cv2.subtract(lab[:,:,2], 15)  # Blue shadows
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Add atmospheric haze
        haze = cv2.GaussianBlur(result, (51, 51), 0)
        result = cv2.addWeighted(result, 0.75, haze, 0.25, 0)
        
        # Increase contrast
        return cv2.convertScaleAbs(result, alpha=1.4, beta=-20)
    
    def ghost_shell_style(self, img_rgb, gray):
        """Ghost in the Shell anime cyberpunk style"""
        # High contrast with blue tint
        result = cv2.convertScaleAbs(img_rgb, alpha=1.3, beta=-10)
        
        # Blue color cast
        result[:,:,2] = np.clip(result[:,:,2] * 1.2, 0, 255)
        
        # Add digital artifacts
        h, w = result.shape[:2]
        for _ in range(10):
            x = np.random.randint(0, w-20)
            y = np.random.randint(0, h-5)
            result[y:y+2, x:x+15] = [0, 100, 255]
        
        return result
    
    def akira_cyberpunk(self, img_rgb, gray):
        """Akira-style cyberpunk with red accents"""
        # High saturation
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.6, 0, 255)
        
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Red accent overlay
        red_mask = (result[:,:,0] > result[:,:,1]) & (result[:,:,0] > result[:,:,2])
        result[red_mask] = cv2.addWeighted(result[red_mask], 0.7, 
                                         np.full_like(result[red_mask], [255, 0, 0]), 0.3, 0)
        
        return cv2.convertScaleAbs(result, alpha=1.2, beta=5)
    
    def neon_noir_style(self, img_rgb, gray):
        """Neon noir with dramatic lighting"""
        # High contrast base
        contrast = cv2.convertScaleAbs(img_rgb, alpha=2.0, beta=-50)
        
        # Neon highlights
        bright_mask = cv2.cvtColor(contrast, cv2.COLOR_RGB2GRAY) > 150
        neon_colors = np.array([[255, 0, 255], [0, 255, 255], [255, 255, 0]])
        
        for i, color in enumerate(neon_colors):
            mask_slice = bright_mask & ((i * 85) < cv2.cvtColor(contrast, cv2.COLOR_RGB2GRAY)) & \
                        (cv2.cvtColor(contrast, cv2.COLOR_RGB2GRAY) < ((i + 1) * 85))
            contrast[mask_slice] = color
        
        # Glow effect
        glow = cv2.GaussianBlur(contrast, (21, 21), 0)
        return cv2.addWeighted(contrast, 0.7, glow, 0.3, 0)
    
    # VINTAGE & RETRO STYLES (161-175)
    
    def film_noir_style(self, img_rgb, gray):
        """Classic film noir with high contrast and dramatic shadows"""
        # Convert to grayscale with high contrast
        noir = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        noir = cv2.convertScaleAbs(noir, alpha=2.5, beta=-80)
        
        # Apply CLAHE for dramatic lighting
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        noir = clahe.apply(noir)
        
        # Add film grain
        h, w = noir.shape
        grain = np.random.normal(0, 20, (h, w)).astype(np.int16)
        noir = np.clip(noir.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        
        # Slight sepia conversion
        result = cv2.cvtColor(noir, cv2.COLOR_GRAY2RGB)
        sepia_kernel = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])
        
        return cv2.transform(result, sepia_kernel).astype(np.uint8)
    
    def art_deco_style(self, img_rgb, gray):
        """Art Deco geometric patterns and gold tones"""
        # Gold color grading
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        lab[:,:,1] = cv2.add(lab[:,:,1], 15)  # Warm tint
        lab[:,:,2] = cv2.add(lab[:,:,2], 10)  # Yellow tint
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Posterize colors for geometric effect
        result = (result // 32) * 32
        
        # Add geometric patterns
        h, w = result.shape[:2]
        for i in range(0, h, 20):
            cv2.line(result, (0, i), (w, i), (255, 215, 0), 1)
        
        return cv2.convertScaleAbs(result, alpha=1.1, beta=10)
    
    def vintage_poster(self, img_rgb, gray):
        """Vintage poster with limited color palette"""
        # Reduce to vintage color palette
        data = img_rgb.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        vintage_colors = centers[labels.flatten()].reshape(img_rgb.shape)
        
        # Add paper texture
        h, w = vintage_colors.shape[:2]
        texture = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
        result = np.clip(vintage_colors.astype(np.int16) + texture, 0, 255).astype(np.uint8)
        
        # Sepia tone
        sepia_kernel = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])
        
        return cv2.transform(result, sepia_kernel).astype(np.uint8)
    
    def pinup_style(self, img_rgb, gray):
        """1950s pin-up art style with soft colors"""
        # Soft, dreamy blur
        soft = cv2.bilateralFilter(img_rgb, 15, 50, 50)
        
        # Enhance skin tones
        lab = cv2.cvtColor(soft, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = cv2.add(lab[:,:,0], 10)  # Brighten
        lab[:,:,1] = cv2.subtract(lab[:,:,1], 5)  # Reduce red
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Add warm glow
        glow = cv2.GaussianBlur(result, (31, 31), 0)
        warm_glow = glow.copy()
        warm_glow[:,:,0] = np.clip(warm_glow[:,:,0] * 1.1, 0, 255)  # Boost red
        warm_glow[:,:,1] = np.clip(warm_glow[:,:,1] * 1.05, 0, 255)  # Slight green
        
        return cv2.addWeighted(result, 0.8, warm_glow, 0.2, 0)
    
    def americana_50s(self, img_rgb, gray):
        """1950s Americana with bright colors"""
        # Boost saturation and contrast
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.4, 0, 255)  # Saturation
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.2, 0, 255)  # Brightness
        
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Red, white, blue emphasis
        red_mask = result[:,:,0] > 150
        blue_mask = result[:,:,2] > 150
        
        result[red_mask, 0] = 255
        result[blue_mask, 2] = 255
        
        return cv2.convertScaleAbs(result, alpha=1.1, beta=5)
    
    def psychedelic_60s(self, img_rgb, gray):
        """1960s psychedelic with swirling colors"""
        # Extreme saturation
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = 255  # Maximum saturation
        
        # Shift hues for psychedelic effect
        h, w = hsv.shape[:2]
        x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        wave = np.sin(x * 10 + y * 10) * 50
        hsv[:,:,0] = np.clip(hsv[:,:,0].astype(np.int16) + wave.astype(np.int16), 0, 179)
        
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add swirl effect
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 3
        
        for i in range(h):
            for j in range(w):
                dx, dy = j - center_x, i - center_y
                distance = np.sqrt(dx*dx + dy*dy)
                if distance < radius:
                    angle = np.arctan2(dy, dx)
                    new_angle = angle + (radius - distance) / radius * 2
                    new_x = int(center_x + distance * np.cos(new_angle))
                    new_y = int(center_y + distance * np.sin(new_angle))
                    if 0 <= new_x < w and 0 <= new_y < h:
                        result[i, j] = img_rgb[new_y, new_x]
        
        return result
    
    def disco_70s_style(self, img_rgb, gray):
        """1970s disco with sparkle and shine"""
        # Gold and purple tones
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        lab[:,:,1] = cv2.add(lab[:,:,1], 20)  # Warm
        lab[:,:,2] = cv2.add(lab[:,:,2], 15)  # Yellow
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Add sparkle effects
        h, w = result.shape[:2]
        sparkles = np.random.random((h, w)) > 0.98
        result[sparkles] = [255, 215, 0]  # Gold sparkles
        
        # Glow effect
        glow = cv2.GaussianBlur(result, (15, 15), 0)
        return cv2.addWeighted(result, 0.8, glow, 0.2, 0)
    
    def neon_80s_style(self, img_rgb, gray):
        """1980s neon with bright pinks and blues"""
        # High contrast and saturation
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 2.0, 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.3, 0, 255)
        
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Neon color emphasis
        pink_mask = (result[:,:,0] > 200) & (result[:,:,2] > 150)
        blue_mask = (result[:,:,2] > 200) & (result[:,:,0] < 100)
        
        result[pink_mask] = [255, 20, 147]  # Hot pink
        result[blue_mask] = [0, 191, 255]   # Deep sky blue
        
        # Glow effect
        glow = cv2.GaussianBlur(result, (21, 21), 0)
        return cv2.addWeighted(result, 0.7, glow, 0.3, 0)
    
    def grunge_90s_style(self, img_rgb, gray):
        """1990s grunge with desaturated colors and texture"""
        # Desaturate
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = hsv[:,:,1] * 0.6  # Reduce saturation
        
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add grunge texture
        h, w = result.shape[:2]
        grunge = np.random.normal(0, 30, (h, w, 3)).astype(np.int16)
        result = np.clip(result.astype(np.int16) + grunge, 0, 255).astype(np.uint8)
        
        # Darken overall
        return cv2.convertScaleAbs(result, alpha=0.8, beta=-20)
    
    def polaroid_vintage(self, img_rgb, gray):
        """Vintage Polaroid photo effect"""
        # Warm, faded look
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = cv2.subtract(lab[:,:,0], 10)  # Slightly darker
        lab[:,:,1] = cv2.add(lab[:,:,1], 8)       # Warm tint
        lab[:,:,2] = cv2.add(lab[:,:,2], 12)      # Yellow cast
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Add border (Polaroid style)
        h, w = result.shape[:2]
        border_size = 20
        bordered = np.full((h + 2*border_size, w + 2*border_size, 3), 255, dtype=np.uint8)
        bordered[border_size:h+border_size, border_size:w+border_size] = result
        
        # Resize back to original
        return cv2.resize(bordered, (w, h))
    
    def retro_futurism(self, img_rgb, gray):
        """Retro futurism style"""
        # Chrome and neon colors
        result = cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=10)
        
        # Metallic sheen
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        metallic = cv2.addWeighted(result, 0.7, gray_3ch, 0.3, 0)
        
        # Add geometric elements
        h, w = metallic.shape[:2]
        for i in range(0, w, 30):
            cv2.line(metallic, (i, 0), (i, h), (255, 255, 255), 1)
        
        return metallic
    
    def mid_century_modern(self, img_rgb, gray):
        """Mid-century modern design aesthetic"""
        # Limited color palette
        data = img_rgb.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        result = centers[labels.flatten()].reshape(img_rgb.shape)
        
        # Clean, geometric look
        return cv2.bilateralFilter(result, 9, 75, 75)
    
    def victorian_gothic(self, img_rgb, gray):
        """Victorian Gothic dark atmosphere"""
        # Dark, moody tones
        result = cv2.convertScaleAbs(img_rgb, alpha=0.7, beta=-30)
        
        # Purple/brown color cast
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        lab[:,:,1] = cv2.add(lab[:,:,1], 10)   # Slight red
        lab[:,:,2] = cv2.subtract(lab[:,:,2], 15)  # Blue tint
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Add vignette
        h, w = result.shape[:2]
        center_x, center_y = w // 2, h // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        vignette = 1 - (distances / max_dist) * 0.6
        vignette = np.clip(vignette, 0.4, 1)
        
        for i in range(3):
            result[:,:,i] = result[:,:,i] * vignette
        
        return result.astype(np.uint8)
    
    def steampunk_style(self, img_rgb, gray):
        """Steampunk brass and copper tones"""
        # Sepia base
        sepia_kernel = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])
        
        result = cv2.transform(img_rgb, sepia_kernel).astype(np.uint8)
        
        # Copper highlights
        bright_mask = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY) > 150
        result[bright_mask] = cv2.addWeighted(result[bright_mask], 0.7, 
                                            np.full_like(result[bright_mask], [184, 115, 51]), 0.3, 0)
        
        # Add mechanical texture
        h, w = result.shape[:2]
        for i in range(0, h, 15):
            for j in range(0, w, 15):
                if np.random.random() > 0.8:
                    cv2.circle(result, (j, i), 2, (139, 69, 19), -1)
        
        return result
    
    def dieselpunk_style(self, img_rgb, gray):
        """Dieselpunk industrial aesthetic"""
        # High contrast with metallic tones
        result = cv2.convertScaleAbs(img_rgb, alpha=1.4, beta=-20)
        
        # Industrial color grading
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        lab[:,:,1] = cv2.subtract(lab[:,:,1], 10)  # Less red
        lab[:,:,2] = cv2.subtract(lab[:,:,2], 20)  # More blue
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Add rust and metal effects
        rust_mask = np.random.random(result.shape[:2]) > 0.95
        result[rust_mask] = [139, 69, 19]  # Rust color
        
        return result
    
    # ANCIENT/OLD STYLES (176-185)
    
    def ancient_manuscript(self, img_rgb, gray):
        """Ancient parchment manuscript style"""
        # Aged parchment color
        sepia_kernel = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])
        
        result = cv2.transform(img_rgb, sepia_kernel).astype(np.uint8)
        
        # Age spots and stains
        h, w = result.shape[:2]
        for _ in range(30):
            x, y = np.random.randint(10, w-10), np.random.randint(10, h-10)
            radius = np.random.randint(3, 15)
            darkness = np.random.randint(50, 150)
            cv2.circle(result, (x, y), radius, (darkness, darkness, darkness), -1)
        
        # Cracked texture
        noise = np.random.normal(0, 20, (h, w, 3)).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return result
    
    def medieval_illumination(self, img_rgb, gray):
        """Medieval illuminated manuscript style"""
        # Gold leaf effect
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        lab[:,:,1] = cv2.add(lab[:,:,1], 25)  # Gold tint
        lab[:,:,2] = cv2.add(lab[:,:,2], 30)  # Yellow
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Add illuminated borders
        h, w = result.shape[:2]
        border_width = 10
        
        # Gold border
        result[:border_width, :] = [255, 215, 0]  # Top
        result[-border_width:, :] = [255, 215, 0]  # Bottom
        result[:, :border_width] = [255, 215, 0]  # Left
        result[:, -border_width:] = [255, 215, 0]  # Right
        
        # Add decorative elements
        for _ in range(5):
            x, y = np.random.randint(20, w-20), np.random.randint(20, h-20)
            cv2.circle(result, (x, y), 8, (255, 215, 0), 2)
        
        return result
    
    def tapestry_weave(self, img_rgb, gray):
        """Medieval tapestry weave texture"""
        # Reduce color palette
        data = img_rgb.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        result = centers[labels.flatten()].reshape(img_rgb.shape)
        
        # Add weave texture
        h, w = result.shape[:2]
        weave = np.zeros((h, w), dtype=np.uint8)
        
        # Horizontal threads
        for i in range(0, h, 2):
            weave[i, :] = 255
        
        # Vertical threads
        for j in range(0, w, 2):
            weave[:, j] = 255
        
        # Apply weave pattern
        weave_3ch = cv2.cvtColor(weave, cv2.COLOR_GRAY2RGB)
        result = cv2.addWeighted(result, 0.8, weave_3ch, 0.2, 0)
        
        return result
    
    def ancient_fresco(self, img_rgb, gray):
        """Ancient fresco wall painting style"""
        # Faded, weathered look
        result = cv2.convertScaleAbs(img_rgb, alpha=0.8, beta=20)
        
        # Earth tone color grading
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        lab[:,:,1] = cv2.add(lab[:,:,1], 15)  # Warm tint
        lab[:,:,2] = cv2.add(lab[:,:,2], 20)  # Yellow/brown
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Add weathering and cracks
        h, w = result.shape[:2]
        
        # Random weathering spots
        for _ in range(50):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            size = np.random.randint(5, 20)
            fade = np.random.randint(30, 80)
            cv2.circle(result, (x, y), size, (fade, fade, fade), -1)
        
        # Add cracks
        for _ in range(10):
            start_x, start_y = np.random.randint(0, w), np.random.randint(0, h)
            end_x, end_y = np.random.randint(0, w), np.random.randint(0, h)
            cv2.line(result, (start_x, start_y), (end_x, end_y), (100, 100, 100), 1)
        
        return result
    
    def byzantine_icon(self, img_rgb, gray):
        """Byzantine religious icon style"""
        # Gold background emphasis
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        lab[:,:,1] = cv2.add(lab[:,:,1], 30)  # Strong warm tint
        lab[:,:,2] = cv2.add(lab[:,:,2], 40)  # Gold yellow
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Posterize for iconic effect
        result = (result // 16) * 16
        
        # Add halo effects
        h, w = result.shape[:2]
        bright_areas = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY) > 180
        
        # Dilate bright areas for halo effect
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        halo_mask = cv2.dilate(bright_areas.astype(np.uint8), kernel)
        
        result[halo_mask > 0] = cv2.addWeighted(result[halo_mask > 0], 0.7, 
                                              np.full_like(result[halo_mask > 0], [255, 215, 0]), 0.3, 0)
        
        return result
    
    def egyptian_style(self, img_rgb, gray):
        """Ancient Egyptian hieroglyphic art style"""
        # High contrast with earth tones
        result = cv2.convertScaleAbs(img_rgb, alpha=1.5, beta=-30)
        
        # Egyptian color palette
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        lab[:,:,1] = cv2.add(lab[:,:,1], 20)  # Warm
        lab[:,:,2] = cv2.add(lab[:,:,2], 25)  # Yellow/orange
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Posterize for flat Egyptian style
        result = (result // 32) * 32
        
        # Add hieroglyphic-style outlines
        edges = cv2.Canny(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY), 50, 150)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8))
        
        result[edges > 0] = [139, 69, 19]  # Brown outlines
        
        return result
    
    def greek_pottery_style(self, img_rgb, gray):
        """Ancient Greek pottery black-figure style"""
        # Convert to high contrast
        result = cv2.convertScaleAbs(img_rgb, alpha=2.0, beta=-50)
        
        # Create pottery color scheme
        pottery_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        
        # Threshold for black figure style
        _, thresh = cv2.threshold(pottery_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Create red-orange pottery base
        pottery = np.full_like(img_rgb, [205, 133, 63])  # Peru color
        
        # Apply black figures
        pottery[thresh == 0] = [0, 0, 0]
        
        # Add pottery texture
        h, w = pottery.shape[:2]
        for i in range(0, h, 5):
            pottery[i, :] = cv2.addWeighted(pottery[i, :], 0.9, 
                                          np.full_like(pottery[i, :], [139, 69, 19]), 0.1, 0)
        
        return pottery
    
    def roman_mosaic(self, img_rgb, gray):
        """Roman mosaic tile effect"""
        # Pixelate for tile effect
        h, w = img_rgb.shape[:2]
        tile_size = 8
        
        # Reduce resolution
        small = cv2.resize(img_rgb, (w//tile_size, h//tile_size), interpolation=cv2.INTER_AREA)
        
        # Resize back up
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Add tile borders
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                if i < h-1 and j < w-1:
                    cv2.rectangle(mosaic, (j, i), (j+tile_size-1, i+tile_size-1), (0, 0, 0), 1)
        
        # Add marble texture
        marble_noise = np.random.normal(0, 10, (h, w, 3)).astype(np.int16)
        mosaic = np.clip(mosaic.astype(np.int16) + marble_noise, 0, 255).astype(np.uint8)
        
        return mosaic
    
    def cave_painting_style(self, img_rgb, gray):
        """Prehistoric cave painting style"""
        # Earth tone palette
        earth_tones = np.array([[139, 69, 19],   # Saddle brown
                               [160, 82, 45],   # Saddle brown
                               [210, 180, 140], # Tan
                               [222, 184, 135], # Burlywood
                               [205, 133, 63],  # Peru
                               [128, 0, 0]])    # Maroon
        
        # Quantize to earth tones
        data = img_rgb.reshape((-1, 3)).astype(np.float32)
        
        # Find closest earth tone for each pixel
        result = np.zeros_like(img_rgb)
        for i in range(img_rgb.shape[0]):
            for j in range(img_rgb.shape[1]):
                pixel = img_rgb[i, j]
                distances = np.sum((earth_tones - pixel)**2, axis=1)
                closest_idx = np.argmin(distances)
                result[i, j] = earth_tones[closest_idx]
        
        # Add rock texture
        h, w = result.shape[:2]
        rock_texture = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
        result = np.clip(result.astype(np.int16) + rock_texture, 0, 255).astype(np.uint8)
        
        return result
    
    def parchment_scroll(self, img_rgb, gray):
        """Ancient parchment scroll effect"""
        # Aged paper color
        sepia_kernel = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])
        
        result = cv2.transform(img_rgb, sepia_kernel).astype(np.uint8)
        
        # Add aging and yellowing
        result = cv2.convertScaleAbs(result, alpha=0.9, beta=30)
        
        # Add wrinkles and creases
        h, w = result.shape[:2]
        
        # Horizontal creases
        for _ in range(5):
            y = np.random.randint(h//4, 3*h//4)
            crease_strength = np.random.randint(20, 40)
            result[y:y+2, :] = cv2.subtract(result[y:y+2, :], crease_strength)
        
        # Add ink blots
        for _ in range(8):
            x, y = np.random.randint(20, w-20), np.random.randint(20, h-20)
            radius = np.random.randint(3, 10)
            cv2.circle(result, (x, y), radius, (101, 67, 33), -1)
        
        return result
    
    # MODERN DIGITAL STYLES (186-200)
    
    def low_poly_style(self, img_rgb, gray):
        """Low polygon geometric art style"""
        # Dramatic color reduction
        data = img_rgb.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        result = centers[labels.flatten()].reshape(img_rgb.shape)
        
        # Add geometric edges
        edges = cv2.Canny(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY), 30, 100)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8))
        
        # Black polygon outlines
        result[edges > 0] = [0, 0, 0]
        
        return result
    
    def pixel_art_style(self, img_rgb, gray):
        """8-bit pixel art effect"""
        h, w = img_rgb.shape[:2]
        
        # Reduce resolution dramatically
        pixel_size = 8
        small = cv2.resize(img_rgb, (w//pixel_size, h//pixel_size), interpolation=cv2.INTER_AREA)
        
        # Quantize colors to 8-bit palette
        small = (small // 32) * 32
        
        # Resize back with nearest neighbor
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return pixelated
    
    def voxel_art_style(self, img_rgb, gray):
        """3D voxel art appearance"""
        # Pixelate first
        h, w = img_rgb.shape[:2]
        voxel_size = 12
        
        small = cv2.resize(img_rgb, (w//voxel_size, h//voxel_size), interpolation=cv2.INTER_AREA)
        voxelized = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Add 3D shading effect
        for i in range(0, h, voxel_size):
            for j in range(0, w, voxel_size):
                if i < h-voxel_size and j < w-voxel_size:
                    # Top face (lighter)
                    voxelized[i:i+voxel_size//3, j:j+voxel_size] = \
                        cv2.convertScaleAbs(voxelized[i:i+voxel_size//3, j:j+voxel_size], 
                                          alpha=1.3, beta=20)
                    
                    # Right face (darker)
                    voxelized[i+voxel_size//3:i+voxel_size, j+2*voxel_size//3:j+voxel_size] = \
                        cv2.convertScaleAbs(voxelized[i+voxel_size//3:i+voxel_size, 
                                                   j+2*voxel_size//3:j+voxel_size], 
                                          alpha=0.7, beta=-20)
        
        return voxelized
    
    def wireframe_style(self, img_rgb, gray):
        """3D wireframe effect"""
        # Edge detection
        edges = cv2.Canny(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), 50, 150)
        
        # Create wireframe on dark background
        wireframe = np.zeros_like(img_rgb)
        
        # Green wireframe lines (classic computer graphics)
        wireframe[edges > 0] = [0, 255, 0]
        
        # Add grid pattern
        h, w = wireframe.shape[:2]
        grid_spacing = 20
        
        # Horizontal lines
        for i in range(0, h, grid_spacing):
            wireframe[i, :] = [0, 150, 0]
        
        # Vertical lines
        for j in range(0, w, grid_spacing):
            wireframe[:, j] = [0, 150, 0]
        
        # Glow effect
        glow = cv2.GaussianBlur(wireframe, (9, 9), 0)
        result = cv2.addWeighted(wireframe, 0.8, glow, 0.4, 0)
        
        return result
    
    def procedural_style(self, img_rgb, gray):
        """Procedural generation pattern"""
        h, w = img_rgb.shape[:2]
        
        # Generate Perlin-like noise pattern
        x, y = np.meshgrid(np.linspace(0, 4, w), np.linspace(0, 4, h))
        
        noise1 = np.sin(x * 2 * np.pi) * np.cos(y * 2 * np.pi)
        noise2 = np.sin(x * 4 * np.pi) * np.cos(y * 4 * np.pi) * 0.5
        noise3 = np.sin(x * 8 * np.pi) * np.cos(y * 8 * np.pi) * 0.25
        
        combined_noise = (noise1 + noise2 + noise3) * 0.5 + 0.5
        
        # Apply noise to image
        result = img_rgb.copy().astype(np.float32)
        for i in range(3):
            result[:,:,i] = result[:,:,i] * combined_noise
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def photogrammetry_style(self, img_rgb, gray):
        """3D scan/photogrammetry reconstruction look"""
        # High contrast with technical appearance
        result = cv2.convertScaleAbs(img_rgb, alpha=1.4, beta=-30)
        
        # Add scan lines
        h, w = result.shape[:2]
        for i in range(0, h, 5):
            result[i, :] = cv2.addWeighted(result[i, :], 0.9, 
                                         np.full_like(result[i, :], [100, 200, 255]), 0.1, 0)
        
        # Add measurement markers
        for _ in range(10):
            x, y = np.random.randint(10, w-10), np.random.randint(10, h-10)
            cv2.circle(result, (x, y), 3, (255, 255, 0), 1)
            cv2.line(result, (x-5, y), (x+5, y), (255, 255, 0), 1)
            cv2.line(result, (x, y-5), (x, y+5), (255, 255, 0), 1)
        
        return result
    
    def neural_style_transfer(self, img_rgb, gray):
        """Neural network style transfer appearance"""
        # Artistic brush-like strokes
        result = cv2.bilateralFilter(img_rgb, 15, 50, 50)
        
        # Add painterly texture
        h, w = result.shape[:2]
        
        # Create brush stroke pattern
        for _ in range(100):
            x, y = np.random.randint(5, w-5), np.random.randint(5, h-5)
            angle = np.random.uniform(0, 2*np.pi)
            length = np.random.randint(5, 15)
            
            end_x = int(x + length * np.cos(angle))
            end_y = int(y + length * np.sin(angle))
            
            if 0 <= end_x < w and 0 <= end_y < h:
                color = result[y, x].astype(int)
                cv2.line(result, (x, y), (end_x, end_y), color.tolist(), 2)
        
        return result
    
    def deep_dream_style(self, img_rgb, gray):
        """Deep Dream hallucinogenic effect"""
        # Enhanced patterns and textures
        result = cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=10)
        
        # Add swirling patterns
        h, w = result.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        for i in range(h):
            for j in range(w):
                dx, dy = j - center_x, i - center_y
                distance = np.sqrt(dx*dx + dy*dy)
                angle = np.arctan2(dy, dx)
                
                # Create swirl effect
                new_angle = angle + (distance / 100) * np.sin(distance / 20)
                wave_x = int(center_x + distance * np.cos(new_angle))
                wave_y = int(center_y + distance * np.sin(new_angle))
                
                if 0 <= wave_x < w and 0 <= wave_y < h:
                    result[i, j] = cv2.addWeighted(result[i, j], 0.7, 
                                                 img_rgb[wave_y, wave_x], 0.3, 0)
        
        # Add fractal-like details
        enhanced = cv2.addWeighted(result, 0.8, 
                                 cv2.bilateralFilter(result, 9, 200, 200), 0.2, 0)
        
        return enhanced
    
    def algorithmic_art(self, img_rgb, gray):
        """Mathematical algorithmic patterns"""
        h, w = img_rgb.shape[:2]
        result = img_rgb.copy()
        
        # Generate mathematical pattern
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Complex mathematical function
        pattern = np.sin(x * 0.1) * np.cos(y * 0.1) + \
                 np.sin(x * 0.05 + y * 0.05) * 0.5
        
        # Normalize pattern
        pattern = (pattern + 1.5) / 3.0
        
        # Apply pattern to image
        for i in range(3):
            result[:,:,i] = result[:,:,i] * pattern
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def generative_art_style(self, img_rgb, gray):
        """Generative art with random elements"""
        result = img_rgb.copy()
        h, w = result.shape[:2]
        
        # Add random geometric shapes
        for _ in range(50):
            shape_type = np.random.choice(['circle', 'rectangle', 'line'])
            color = [np.random.randint(0, 255) for _ in range(3)]
            
            if shape_type == 'circle':
                center = (np.random.randint(0, w), np.random.randint(0, h))
                radius = np.random.randint(5, 30)
                cv2.circle(result, center, radius, color, -1)
            elif shape_type == 'rectangle':
                pt1 = (np.random.randint(0, w), np.random.randint(0, h))
                pt2 = (np.random.randint(0, w), np.random.randint(0, h))
                cv2.rectangle(result, pt1, pt2, color, -1)
            else:  # line
                pt1 = (np.random.randint(0, w), np.random.randint(0, h))
                pt2 = (np.random.randint(0, w), np.random.randint(0, h))
                cv2.line(result, pt1, pt2, color, np.random.randint(1, 5))
        
        # Blend with original
        return cv2.addWeighted(img_rgb, 0.6, result, 0.4, 0)
    
    def glitch_datamosh(self, img_rgb, gray):
        """Datamoshing glitch effect"""
        result = img_rgb.copy()
        h, w, c = result.shape
        
        # Random pixel displacement
        for _ in range(20):
            # Random block coordinates
            x1, y1 = np.random.randint(0, w-50), np.random.randint(0, h-50)
            x2, y2 = np.random.randint(0, w-50), np.random.randint(0, h-50)
            block_w, block_h = np.random.randint(10, 50), np.random.randint(10, 50)
            
            # Copy block to random location
            if x1+block_w < w and y1+block_h < h and x2+block_w < w and y2+block_h < h:
                result[y2:y2+block_h, x2:x2+block_w] = result[y1:y1+block_h, x1:x1+block_w]
        
        # Add RGB channel corruption
        corruption_mask = np.random.random((h, w, 3)) > 0.97
        result[corruption_mask] = np.random.randint(0, 255, np.sum(corruption_mask))
        
        return result
    
    def chromatic_aberration(self, img_rgb, gray):
        """Lens chromatic aberration effect"""
        h, w = img_rgb.shape[:2]
        result = np.zeros_like(img_rgb)
        
        # Separate RGB channels
        r, g, b = cv2.split(img_rgb)
        
        # Create displacement maps
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        
        # Radial distance from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Scale displacement by distance from center
        displacement = (distance / max_distance) * 3
        
        # Apply different displacements to each channel
        map_x = x + displacement
        map_y = y
        
        # Red channel - shift outward
        map_x_r = np.clip(map_x + 2, 0, w-1).astype(np.float32)
        map_y_r = np.clip(map_y, 0, h-1).astype(np.float32)
        r_shifted = cv2.remap(r, map_x_r, map_y_r, cv2.INTER_LINEAR)
        
        # Blue channel - shift inward
        map_x_b = np.clip(map_x - 2, 0, w-1).astype(np.float32)
        map_y_b = np.clip(map_y, 0, h-1).astype(np.float32)
        b_shifted = cv2.remap(b, map_x_b, map_y_b, cv2.INTER_LINEAR)
        
        # Green channel - no shift
        g_shifted = g
        
        # Combine channels
        result = cv2.merge([r_shifted, g_shifted, b_shifted])
        
        return result
    
    def scan_lines_effect(self, img_rgb, gray):
        """CRT monitor scan lines effect"""
        result = img_rgb.copy()
        h, w = result.shape[:2]
        
        # Add horizontal scan lines
        for i in range(0, h, 2):
            if i < h:
                result[i, :] = cv2.convertScaleAbs(result[i, :], alpha=0.8, beta=0)
        
        # Add slight green tint (old monitor phosphor)
        result[:,:,1] = np.clip(result[:,:,1] * 1.05, 0, 255)
        
        # Add noise
        noise = np.random.normal(0, 5, (h, w, 3)).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return result
    
    def crt_monitor_effect(self, img_rgb, gray):
        """Old CRT monitor display effect"""
        result = img_rgb.copy()
        h, w = result.shape[:2]
        
        # Barrel distortion (curved screen effect)
        center_x, center_y = w // 2, h // 2
        
        # Create distortion maps
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                # Normalize coordinates
                x_norm = (j - center_x) / center_x
                y_norm = (i - center_y) / center_y
                
                # Apply barrel distortion
                r2 = x_norm**2 + y_norm**2
                distortion = 1 + 0.1 * r2
                
                map_x[i, j] = center_x + x_norm * center_x * distortion
                map_y[i, j] = center_y + y_norm * center_y * distortion
        
        # Apply distortion
        result = cv2.remap(img_rgb, map_x, map_y, cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        # Add scan lines and phosphor glow
        for i in range(0, h, 3):
            if i < h:
                result[i, :] = cv2.convertScaleAbs(result[i, :], alpha=0.7, beta=0)
        
        # Green phosphor tint
        result[:,:,1] = np.clip(result[:,:,1] * 1.1, 0, 255)
        
        return result
    def arabic_calligraphy(self, img_rgb, gray):
        """Arabic calligraphy style with flowing lines"""
        # Create flowing, calligraphic lines
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to create thicker strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        strokes = cv2.dilate(edges, kernel, iterations=2)
        
        # Create paper texture
        h, w = gray.shape
        paper = np.ones((h, w, 3), dtype=np.uint8) * 245
        paper_texture = np.random.normal(0, 5, (h, w, 3))
        paper = np.clip(paper + paper_texture, 230, 255).astype(np.uint8)
        
        # Apply calligraphy strokes
        calligraphy = paper.copy()
        stroke_mask = strokes == 255
        calligraphy[stroke_mask] = [20, 15, 10]  # Dark brown ink
        
        return calligraphy

    def islamic_geometric(self, img_rgb, gray):
        """Islamic geometric pattern overlay"""
        h, w = gray.shape
        geometric = img_rgb.copy()
        
        # Create geometric pattern
        pattern = np.ones((h, w), dtype=np.uint8) * 255
        
        # Eight-pointed star pattern
        for y in range(0, h, 40):
            for x in range(0, w, 40):
                if x + 40 < w and y + 40 < h:
                    # Draw star pattern
                    center = (x + 20, y + 20)
                    pts = []
                    for i in range(16):
                        angle = i * np.pi / 8
                        radius = 15 if i % 2 == 0 else 8
                        px = int(center[0] + radius * np.cos(angle))
                        py = int(center[1] + radius * np.sin(angle))
                        pts.append([px, py])
                    
                    pts = np.array(pts, np.int32)
                    cv2.fillPoly(pattern, [pts], 0)
        
        # Blend with image
        pattern_3d = cv2.cvtColor(pattern, cv2.COLOR_GRAY2RGB)
        result = cv2.addWeighted(geometric, 0.7, pattern_3d, 0.3, 0)
        
        return result

    def arabesque_pattern(self, img_rgb, gray):
        """Flowing arabesque vine patterns"""
        h, w = gray.shape
        arabesque = img_rgb.copy()
        
        # Create flowing vine pattern
        overlay = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Draw flowing curves
        num_vines = 8
        for _ in range(num_vines):
            start_x = np.random.randint(0, w)
            start_y = np.random.randint(0, h)
            
            points = [(start_x, start_y)]
            for i in range(20):
                angle = i * 0.3 + np.random.uniform(-0.5, 0.5)
                step = 15
                new_x = int(points[-1][0] + step * np.cos(angle))
                new_y = int(points[-1][1] + step * np.sin(angle))
                
                if 0 <= new_x < w and 0 <= new_y < h:
                    points.append((new_x, new_y))
                    cv2.line(overlay, points[-2], points[-1], [100, 150, 100], 3)
                    
                    # Add leaves
                    if i % 3 == 0:
                        leaf_pts = np.array([
                            [new_x, new_y],
                            [new_x + 8, new_y - 5],
                            [new_x + 12, new_y],
                            [new_x + 8, new_y + 5]
                        ])
                        cv2.fillPoly(overlay, [leaf_pts], [80, 120, 80])
        
        result = cv2.addWeighted(arabesque, 0.8, overlay, 0.2, 0)
        return result

    def mosque_architecture(self, img_rgb, gray):
        """Mosque architectural style with domes and minarets"""
        # Strong edge detection
        edges = cv2.Canny(gray, 30, 100)
        
        # Create architectural effect
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        arch_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Enhance contrast for architectural feel
        enhanced = cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=10)
        
        # Add golden tint
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = np.clip(hsv[:,:,0] + 15, 0, 179)  # Shift to golden hue
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Overlay architectural lines
        arch_mask = arch_edges == 255
        result[arch_mask] = [139, 115, 85]  # Sandy brown
        
        return result

    def persian_miniature(self, img_rgb, gray):
        """Persian miniature painting style"""
        # Reduce image to flat colors
        miniature = cv2.bilateralFilter(img_rgb, 20, 80, 80)
        
        # Enhance saturation
        hsv = cv2.cvtColor(miniature, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.5)  # High saturation
        miniature = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add gold leaf effect
        brightness = cv2.cvtColor(miniature, cv2.COLOR_RGB2GRAY)
        bright_areas = brightness > 180
        miniature[bright_areas] = [255, 215, 0]  # Gold
        
        # Add fine border lines
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        edge_mask = edges == 255
        miniature[edge_mask] = [139, 69, 19]  # Dark brown outline
        
        return miniature

    def ottoman_art(self, img_rgb, gray):
        """Ottoman artistic style with rich colors"""
        # Apply strong bilateral filter
        ottoman = cv2.bilateralFilter(img_rgb, 25, 100, 100)
        
        # Enhance blues and reds (Ottoman colors)
        hsv = cv2.cvtColor(ottoman, cv2.COLOR_RGB2HSV)
        
        # Enhance blue regions
        blue_mask = (hsv[:,:,0] >= 100) & (hsv[:,:,0] <= 130)
        hsv[blue_mask, 1] = np.clip(hsv[blue_mask, 1] * 1.4, 0, 255)
        
        # Enhance red regions
        red_mask = (hsv[:,:,0] <= 10) | (hsv[:,:,0] >= 170)
        hsv[red_mask, 1] = np.clip(hsv[red_mask, 1] * 1.3, 0, 255)
        
        ottoman = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add ornate texture
        h, w = gray.shape
        texture = np.random.randint(-5, 5, (h, w, 3), dtype=np.int16)
        ottoman = np.clip(ottoman.astype(np.int16) + texture, 0, 255).astype(np.uint8)
        
        return ottoman

    def mamluk_style(self, img_rgb, gray):
        """Mamluk metalwork and manuscript style"""
        # Create metallic effect
        mamluk = cv2.convertScaleAbs(img_rgb, alpha=0.8, beta=20)
        
        # Add copper/bronze tint
        hsv = cv2.cvtColor(mamluk, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = np.clip(hsv[:,:,0] + 10, 0, 179)  # Shift to copper
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255)  # Increase saturation
        mamluk = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add intricate line work
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        detailed_edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
        
        edge_mask = detailed_edges == 255
        mamluk[edge_mask] = [101, 67, 33]  # Dark bronze
        
        return mamluk

    def moorish_design(self, img_rgb, gray):
        """Moorish/Andalusian decorative style"""
        # Apply artistic filter
        moorish = cv2.edgePreservingFilter(img_rgb, flags=2, sigma_s=50, sigma_r=0.4)
        
        # Add warm, earthy tones
        hsv = cv2.cvtColor(moorish, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = np.clip(hsv[:,:,0] + 5, 0, 179)  # Warm shift
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255)  # Brighten
        moorish = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add decorative pattern overlay
        h, w = gray.shape
        pattern = np.zeros((h, w), dtype=np.uint8)
        
        # Hexagonal pattern
        for y in range(0, h, 30):
            for x in range(0, w, 30):
                if x + 25 < w and y + 25 < h:
                    center = (x + 15, y + 15)
                    for i in range(6):
                        angle1 = i * np.pi / 3
                        angle2 = (i + 1) * np.pi / 3
                        pt1 = (int(center[0] + 10 * np.cos(angle1)),
                            int(center[1] + 10 * np.sin(angle1)))
                        pt2 = (int(center[0] + 10 * np.cos(angle2)),
                            int(center[1] + 10 * np.sin(angle2)))
                        cv2.line(pattern, pt1, pt2, 255, 1)
        
        # Blend pattern
        pattern_3d = cv2.cvtColor(pattern, cv2.COLOR_GRAY2RGB)
        result = cv2.addWeighted(moorish, 0.9, pattern_3d, 0.1, 0)
        
        return result

    def kufic_script(self, img_rgb, gray):
        """Angular Kufic calligraphy style"""
        # Create angular, geometric strokes
        edges = cv2.Canny(gray, 80, 160)
        
        # Make strokes more angular
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=2)
        
        angular = cv2.bitwise_or(h_lines, v_lines)
        
        # Create parchment background
        h, w = gray.shape
        parchment = np.ones((h, w, 3), dtype=np.uint8) * 240
        noise = np.random.normal(0, 8, (h, w, 3))
        parchment = np.clip(parchment + noise, 220, 255).astype(np.uint8)
        
        # Apply angular script
        script_mask = angular == 255
        parchment[script_mask] = [40, 30, 20]  # Dark brown ink
        
        return parchment

    def nastaliq_calligraphy(self, img_rgb, gray):
        """Flowing Nastaliq script style"""
        # Create flowing, curved lines
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 50, 120)
        
        # Enhance flowing nature
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
        flowing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Create aged paper
        h, w = gray.shape
        paper = np.ones((h, w, 3), dtype=np.uint8) * 250
        
        # Add aging spots
        for _ in range(200):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            size = np.random.randint(2, 8)
            cv2.circle(paper, (x, y), size, [220, 200, 180], -1)
        
        # Apply script
        script_mask = flowing == 255
        paper[script_mask] = [25, 35, 45]  # Blue-black ink
        
        return paper

    def mihrab_style(self, img_rgb, gray):
        """Mihrab (prayer niche) architectural style"""
        # Apply symmetrical enhancement
        h, w = gray.shape
        mihrab = img_rgb.copy()
        
        # Create arch overlay
        arch = np.zeros((h, w), dtype=np.uint8)
        
        # Draw pointed arch
        center_x = w // 2
        arch_width = min(w // 3, 100)
        arch_height = min(h // 2, 150)
        
        # Draw arch outline
        cv2.ellipse(arch, (center_x, h // 2), (arch_width, arch_height), 0, 180, 360, 255, 3)
        
        # Add decorative elements
        for i in range(5):
            y_pos = h // 2 + i * 20
            if y_pos < h - 10:
                cv2.line(arch, (center_x - 30, y_pos), (center_x + 30, y_pos), 255, 2)
        
        # Apply arch pattern
        arch_3d = cv2.cvtColor(arch, cv2.COLOR_GRAY2RGB)
        arch_3d[arch == 255] = [184, 134, 11]  # Gold
        
        result = cv2.addWeighted(mihrab, 0.8, arch_3d, 0.2, 0)
        return result

    def islamic_tilework(self, img_rgb, gray):
        """Islamic ceramic tile patterns"""
        # Posterize colors like ceramic tiles
        tilework = cv2.bilateralFilter(img_rgb, 20, 100, 100)
        
        # Quantize colors
        data = tilework.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        tilework = segmented.reshape(img_rgb.shape)
        
        # Add tile grout lines
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((3, 3), np.uint8)
        grout = cv2.dilate(edges, kernel, iterations=1)
        
        grout_mask = grout == 255
        tilework[grout_mask] = [220, 220, 220]  # Light grout
        
        return tilework

    def crescent_moon_art(self, img_rgb, gray):
        """Islamic crescent moon motif"""
        h, w = gray.shape
        crescent = img_rgb.copy()
        
        # Add crescent and star overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw crescent moon
        center = (w // 2, h // 3)
        radius = min(w, h) // 8
        
        # Full circle
        cv2.circle(overlay, center, radius, [255, 215, 0], -1)
        
        # Subtract smaller circle to create crescent
        offset_center = (center[0] + radius // 3, center[1])
        cv2.circle(overlay, offset_center, radius - 5, [0, 0, 0], -1)
        
        # Add star
        star_center = (center[0] + radius + 20, center[1] - radius // 2)
        star_points = []
        for i in range(10):
            angle = i * np.pi / 5
            radius_star = 15 if i % 2 == 0 else 7
            x = int(star_center[0] + radius_star * np.cos(angle))
            y = int(star_center[1] + radius_star * np.sin(angle))
            star_points.append([x, y])
        
        star_points = np.array(star_points, np.int32)
        cv2.fillPoly(overlay, [star_points], [255, 215, 0])
        
        result = cv2.addWeighted(crescent, 0.7, overlay, 0.3, 0)
        return result

    def star_pattern(self, img_rgb, gray):
        """Eight-pointed Islamic star pattern"""
        h, w = gray.shape
        stars = img_rgb.copy()
        
        # Create star pattern overlay
        pattern = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw eight-pointed stars
        for y in range(0, h, 60):
            for x in range(0, w, 60):
                if x + 50 < w and y + 50 < h:
                    center = (x + 30, y + 30)
                    
                    # Create 8-pointed star
                    star_pts = []
                    for i in range(16):
                        angle = i * np.pi / 8
                        radius = 20 if i % 2 == 0 else 10
                        px = int(center[0] + radius * np.cos(angle))
                        py = int(center[1] + radius * np.sin(angle))
                        star_pts.append([px, py])
                    
                    star_pts = np.array(star_pts, np.int32)
                    cv2.fillPoly(pattern, [star_pts], [100, 149, 237])
        
        result = cv2.addWeighted(stars, 0.8, pattern, 0.2, 0)
        return result

    def islamic_border(self, img_rgb, gray):
        """Decorative Islamic border pattern"""
        h, w = gray.shape
        bordered = img_rgb.copy()
        
        # Create border
        border_width = min(w, h) // 20
        
        # Top and bottom borders
        for y in range(border_width):
            for x in range(w):
                pattern_val = int(127 + 127 * np.sin(x * 0.1))
                bordered[y, x] = [pattern_val, 140, 80]
                bordered[h-1-y, x] = [pattern_val, 140, 80]
        
        # Left and right borders
        for x in range(border_width):
            for y in range(h):
                pattern_val = int(127 + 127 * np.sin(y * 0.1))
                bordered[y, x] = [pattern_val, 140, 80]
                bordered[y, w-1-x] = [pattern_val, 140, 80]
        
        return bordered

    def quranic_manuscript(self, img_rgb, gray):
        """Quranic manuscript style with illumination"""
        # Create aged parchment effect
        manuscript = cv2.convertScaleAbs(img_rgb, alpha=0.7, beta=40)
        
        # Add warm, aged tone
        hsv = cv2.cvtColor(manuscript, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = np.clip(hsv[:,:,0] + 15, 0, 179)  # Warm shift
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 0.8, 0, 255)  # Reduce saturation
        manuscript = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add gold illumination
        bright_areas = cv2.cvtColor(manuscript, cv2.COLOR_RGB2GRAY) > 150
        manuscript[bright_areas] = [255, 215, 0]  # Gold leaf
        
        # Add calligraphic lines
        edges = cv2.Canny(gray, 50, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        callig_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        line_mask = callig_lines == 255
        manuscript[line_mask] = [75, 0, 130]  # Dark purple ink
        
        return manuscript

    def madrasa_style(self, img_rgb, gray):
        """Traditional Islamic school architectural style"""
        # Create scholarly, architectural atmosphere
        madrasa = cv2.bilateralFilter(img_rgb, 15, 50, 50)
        
        # Add warm, studious lighting
        hsv = cv2.cvtColor(madrasa, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = np.clip(hsv[:,:,0] + 10, 0, 179)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255)
        madrasa = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add architectural details
        edges = cv2.Canny(gray, 80, 160)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        arch_details = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        detail_mask = arch_details == 255
        madrasa[detail_mask] = [160, 82, 45]  # Sandstone color
        
        return madrasa

    def islamic_garden(self, img_rgb, gray):
        """Paradise garden (Jannah) style"""
        # Enhance greens and blues
        garden = cv2.bilateralFilter(img_rgb, 15, 80, 80)
        
        hsv = cv2.cvtColor(garden, cv2.COLOR_RGB2HSV)
        
        # Enhance green areas
        green_mask = (hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 85)
        hsv[green_mask, 1] = np.clip(hsv[green_mask, 1] * 1.3, 0, 255)
        hsv[green_mask, 2] = np.clip(hsv[green_mask, 2] * 1.1, 0, 255)
        
        # Enhance blue areas (water)
        blue_mask = (hsv[:,:,0] >= 100) & (hsv[:,:,0] <= 130)
        hsv[blue_mask, 1] = np.clip(hsv[blue_mask, 1] * 1.4, 0, 255)
        
        garden = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add paradise-like glow
        glow = cv2.GaussianBlur(garden, (15, 15), 0)
        result = cv2.addWeighted(garden, 0.8, glow, 0.2, 0)
        
        return result

    def sufi_mystical(self, img_rgb, gray):
        """Sufi mystical art style"""
        # Create ethereal, spiritual effect
        mystical = cv2.bilateralFilter(img_rgb, 20, 100, 100)
        
        # Add spiritual glow
        glow = cv2.GaussianBlur(mystical, (21, 21), 0)
        mystical = cv2.addWeighted(mystical, 0.7, glow, 0.3, 0)
        
        # Add swirling patterns
        h, w = gray.shape
        for i in range(5):
            center = (np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4))
            for radius in range(10, 50, 10):
                for angle in range(0, 360, 30):
                    x = int(center[0] + radius * np.cos(np.radians(angle)))
                    y = int(center[1] + radius * np.sin(np.radians(angle)))
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(mystical, (x, y), 2, [255, 255, 255], 1)
        
        return mystical

    def hajj_pilgrimage(self, img_rgb, gray):
        """Hajj pilgrimage theme with white and gold"""
        # Create pure, pilgrimage atmosphere
        pilgrimage = cv2.convertScaleAbs(img_rgb, alpha=1.1, beta=20)
        
        # Enhance whites (ihram clothing)
        hsv = cv2.cvtColor(pilgrimage, cv2.COLOR_RGB2HSV)
        bright_mask = hsv[:,:,2] > 200
        pilgrimage[bright_mask] = [255, 255, 255]
        
        # Add golden highlights (Kaaba)
        mid_bright = (hsv[:,:,2] > 150) & (hsv[:,:,2] <= 200)
        pilgrimage[mid_bright] = [255, 215, 0]
        
        # Add subtle radial pattern (representing unity)
        h, w = gray.shape
        center = (w // 2, h // 2)
        y, x = np.ogrid[:h, :w]
        mask = (x - center[0])**2 + (y - center[1])**2
        mask = mask / mask.max()
        
        # Subtle circular enhancement
        for i in range(h):
            for j in range(w):
                distance = mask[i, j]
                if distance < 0.7:
                    enhance = 1.0 + 0.1 * (0.7 - distance)
                    pilgrimage[i, j] = np.clip(pilgrimage[i, j] * enhance, 0, 255)
        
        return pilgrimage

    def oil_paint_filter(self, img_rgb, gray):
        """Photoshop Oil Paint filter effect"""
        # Strong bilateral filter for oil paint look
        oil = cv2.bilateralFilter(img_rgb, 25, 100, 100)
        oil = cv2.bilateralFilter(oil, 25, 100, 100)  # Double pass
        
        # Enhance saturation for oil paint vibrancy
        hsv = cv2.cvtColor(oil, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.4)
        oil = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add brush stroke texture
        kernel = np.array([[-1,-1,-1,0,0], [-1,-1,0,1,1], [-1,0,0,1,1], [0,1,1,1,1], [0,1,1,1,1]])
        textured = cv2.filter2D(oil, -1, kernel)
        
        return np.clip(textured, 0, 255).astype(np.uint8)

    def dry_brush_filter(self, img_rgb, gray):
        """Photoshop Dry Brush artistic filter"""
        # Reduce colors to simulate dry brush
        dry = cv2.bilateralFilter(img_rgb, 15, 50, 50)
        
        # Quantize colors
        data = dry.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 12, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        dry = segmented.reshape(img_rgb.shape)
        
        # Add dry brush texture
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7))
        brush_strokes = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        stroke_mask = brush_strokes == 255
        dry[stroke_mask] = dry[stroke_mask] * 0.7
        
        return dry.astype(np.uint8)

    def palette_knife_filter(self, img_rgb, gray):
        """Photoshop Palette Knife filter"""
        # Strong edge-preserving filter
        palette = cv2.edgePreservingFilter(img_rgb, flags=2, sigma_s=100, sigma_r=0.3)
        
        # Add palette knife texture
        kernel_h = np.array([[-1, -2, 0, 2, 1]])
        kernel_v = np.array([[-1], [-2], [0], [2], [1]])
        
        h_texture = cv2.filter2D(palette, -1, kernel_h)
        v_texture = cv2.filter2D(palette, -1, kernel_v)
        
        # Combine textures
        texture = np.sqrt(h_texture.astype(np.float32)**2 + v_texture.astype(np.float32)**2)
        texture = np.clip(texture, 0, 255).astype(np.uint8)
        
        return cv2.addWeighted(palette, 0.7, texture, 0.3, 0)

    def watercolor_filter(self, img_rgb, gray):
        """Photoshop Watercolor artistic filter"""
        # Multiple blur passes for watercolor bleeding
        watercolor = cv2.GaussianBlur(img_rgb, (5, 5), 0)
        watercolor = cv2.bilateralFilter(watercolor, 20, 80, 80)
        
        # Reduce saturation slightly
        hsv = cv2.cvtColor(watercolor, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 0.8)
        watercolor = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add paper texture
        h, w = gray.shape
        paper_texture = np.random.normal(0, 3, (h, w, 3))
        watercolor = np.clip(watercolor.astype(np.float32) + paper_texture, 0, 255).astype(np.uint8)
        
        # Add watercolor bleeding effect
        edges = cv2.Canny(gray, 30, 80)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bleeding = cv2.dilate(edges, kernel, iterations=2)
        
        bleed_mask = bleeding == 255
        watercolor[bleed_mask] = cv2.GaussianBlur(watercolor, (15, 15), 0)[bleed_mask]
        
        return watercolor

    def sponge_filter(self, img_rgb, gray):
        """Photoshop Sponge artistic filter"""
        # Create sponge-like texture
        sponge = cv2.bilateralFilter(img_rgb, 20, 50, 50)
        
        # Add random sponge texture
        h, w = gray.shape
        for _ in range(1000):
            x = np.random.randint(0, w-3)
            y = np.random.randint(0, h-3)
            size = np.random.randint(2, 5)
            
            # Random circular sponge marks
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (x, y), size, 255, -1)
            
            # Apply random color variation
            variation = np.random.randint(-20, 20, 3)
            sponge_mask = mask == 255
            sponge[sponge_mask] = np.clip(sponge[sponge_mask].astype(np.int16) + variation, 0, 255)
        
        return sponge.astype(np.uint8)

    def poster_edges(self, img_rgb, gray):
        """Photoshop Poster Edges filter"""
        # Reduce colors dramatically
        poster = cv2.bilateralFilter(img_rgb, 20, 100, 100)
        
        # Quantize to poster-like colors
        data = poster.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        poster = segmented.reshape(img_rgb.shape)
        
        # Add strong edges
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((3, 3), np.uint8)
        thick_edges = cv2.dilate(edges, kernel, iterations=2)
        
        edge_mask = thick_edges == 255
        poster[edge_mask] = [0, 0, 0]  # Black edges
        
        return poster

    def cutout_filter(self, img_rgb, gray):
        """Photoshop Cutout filter"""
        # Extreme color reduction
        cutout = cv2.bilateralFilter(img_rgb, 25, 150, 150)
        
        # Very aggressive quantization
        data = cutout.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        cutout = segmented.reshape(img_rgb.shape)
        
        # Sharp edges
        edges = cv2.Canny(gray, 150, 250)
        kernel = np.ones((2, 2), np.uint8)
        sharp_edges = cv2.dilate(edges, kernel, iterations=1)
        
        edge_mask = sharp_edges == 255
        cutout[edge_mask] = [20, 20, 20]
        
        return cutout

    def torn_edges(self, img_rgb, gray):
        """Photoshop Torn Edges filter"""
        # Create torn paper effect
        h, w = gray.shape
        torn = img_rgb.copy()
        
        # Create irregular mask for torn edges
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Tear top and bottom edges
        for x in range(w):
            tear_top = int(h * 0.1 + np.random.randint(-h//20, h//20))
            tear_bottom = int(h * 0.9 + np.random.randint(-h//20, h//20))
            mask[:tear_top, x] = 0
            mask[tear_bottom:, x] = 0
        
        # Tear left and right edges
        for y in range(h):
            tear_left = int(w * 0.1 + np.random.randint(-w//20, w//20))
            tear_right = int(w * 0.9 + np.random.randint(-w//20, w//20))
            mask[y, :tear_left] = 0
            mask[y, tear_right:] = 0
        
        # Apply mask
        torn_mask = mask == 0
        torn[torn_mask] = [240, 235, 220]  # Paper color
        
        # Add shadow to torn edges
        edges = cv2.Canny(mask, 100, 200)
        kernel = np.ones((5, 5), np.uint8)
        shadow_edges = cv2.dilate(edges, kernel, iterations=2)
        
        shadow_mask = shadow_edges == 255
        torn[shadow_mask] = torn[shadow_mask] * 0.7
        
        return torn.astype(np.uint8)

    def rough_pastels(self, img_rgb, gray):
        """Photoshop Rough Pastels filter"""
        # Soft pastel effect with rough texture
        pastels = cv2.GaussianBlur(img_rgb, (7, 7), 0)
        pastels = cv2.bilateralFilter(pastels, 15, 50, 50)
        
        # Reduce contrast for soft look
        pastels = cv2.convertScaleAbs(pastels, alpha=0.8, beta=30)
        
        # Add rough paper texture
        h, w = gray.shape
        rough_texture = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
        
        # Make texture rougher in some areas
        for _ in range(500):
            x = np.random.randint(0, w-10)
            y = np.random.randint(0, h-10)
            rough_texture[y:y+10, x:x+10] += np.random.randint(-10, 10)
        
        pastels = np.clip(pastels.astype(np.int16) + rough_texture, 0, 255).astype(np.uint8)
        
        return pastels

    def smudge_stick(self, img_rgb, gray):
        """Photoshop Smudge Stick filter"""
        # Create smudged, blended effect
        smudged = cv2.bilateralFilter(img_rgb, 20, 100, 100)
        
        # Apply directional smudging
        h, w = gray.shape
        
        # Horizontal smudging
        kernel_h = np.ones((1, 15), np.float32) / 15
        h_smudge = cv2.filter2D(smudged, -1, kernel_h)
        
        # Vertical smudging
        kernel_v = np.ones((15, 1), np.float32) / 15
        v_smudge = cv2.filter2D(smudged, -1, kernel_v)
        
        # Combine based on edge direction
        edges_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edges_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        edge_strength_h = np.abs(edges_h)
        edge_strength_v = np.abs(edges_v)
        
        # Use horizontal smudge where vertical edges are strong
        mask_h = edge_strength_v > edge_strength_h
        result = smudged.copy()
        result[mask_h] = h_smudge[mask_h]
        
        # Use vertical smudge where horizontal edges are strong
        mask_v = edge_strength_h > edge_strength_v
        result[mask_v] = v_smudge[mask_v]
        
        return result

    def angled_strokes(self, img_rgb, gray):
        """Photoshop Angled Strokes filter"""
        # Create angled brush strokes
        angled = cv2.bilateralFilter(img_rgb, 15, 80, 80)
        
        # Apply angled convolution kernels
        angle = 45  # degrees
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        
        # Create angled kernel
        kernel_size = 15
        kernel = np.zeros((kernel_size, kernel_size), np.float32)
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                # Distance from center along angled line
                x = j - center
                y = i - center
                dist = abs(x * sin_a - y * cos_a)
                
                if dist < 2:  # Width of stroke
                    kernel[i, j] = 1.0 / (kernel_size * 4)
        
        # Apply angled strokes
        stroked = cv2.filter2D(angled, -1, kernel)
        
        # Blend with original
        result = cv2.addWeighted(angled, 0.6, stroked, 0.4, 0)
        
        return result

    def crosshatch_filter(self, img_rgb, gray):
        """Photoshop Crosshatch filter"""
        h, w = gray.shape
        crosshatch = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Create crosshatch pattern based on intensity
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                if i + 8 < h and j + 8 < w:
                    region = gray[i:i+8, j:j+8]
                    intensity = region.mean() / 255.0
                    
                    # Draw crosshatch lines based on intensity
                    if intensity < 0.9:
                        cv2.line(crosshatch, (j, i), (j+8, i+8), [0, 0, 0], 1)
                    if intensity < 0.7:
                        cv2.line(crosshatch, (j+8, i), (j, i+8), [0, 0, 0], 1)
                    if intensity < 0.5:
                        cv2.line(crosshatch, (j, i+4), (j+8, i+4), [0, 0, 0], 1)
                    if intensity < 0.3:
                        cv2.line(crosshatch, (j+4, i), (j+4, i+8), [0, 0, 0], 1)
        
        # Blend with color information
        color_reduced = cv2.bilateralFilter(img_rgb, 20, 100, 100)
        
        # Apply crosshatch as overlay
        gray_3d = cv2.cvtColor(crosshatch[:,:,0], cv2.COLOR_GRAY2RGB)
        result = cv2.multiply(color_reduced.astype(np.float32), gray_3d.astype(np.float32) / 255.0)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def dark_strokes(self, img_rgb, gray):
        """Photoshop Dark Strokes filter"""
        # Emphasize dark areas with brush strokes
        dark = cv2.bilateralFilter(img_rgb, 15, 50, 50)
        
        # Find dark regions
        dark_mask = gray < 100
        
        # Apply stroke texture to dark areas
        kernel = np.array([[-1, -1, 0, 1, 1],
                        [-1, -1, 0, 1, 1],
                        [-1, -1, 0, 1, 1],
                        [-1, -1, 0, 1, 1],
                        [-1, -1, 0, 1, 1]], dtype=np.float32) / 10
        
        strokes = cv2.filter2D(dark, -1, kernel)
        
        # Apply strokes only to dark areas
        result = dark.copy()
        result[dark_mask] = strokes[dark_mask]
        
        return result

    def ink_outlines(self, img_rgb, gray):
        """Photoshop Ink Outlines filter"""
        # Strong edge detection for ink outlines
        edges = cv2.Canny(gray, 50, 150)
        
        # Thicken edges
        kernel = np.ones((3, 3), np.uint8)
        thick_edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Create white background
        h, w = gray.shape
        ink = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Apply black ink outlines
        edge_mask = thick_edges == 255
        ink[edge_mask] = [0, 0, 0]
        
        # Add some color from original in light areas
        light_areas = gray > 180
        color_reduced = cv2.bilateralFilter(img_rgb, 20, 100, 100)
        ink[light_areas] = color_reduced[light_areas] * 0.3 + ink[light_areas] * 0.7
        
        return ink.astype(np.uint8)

    def spatter_effect(self, img_rgb, gray):
        """Photoshop Spatter filter"""
        # Create paint spatter effect
        spatter = cv2.bilateralFilter(img_rgb, 15, 80, 80)
        
        h, w = gray.shape
        
        # Add random paint splatters
        for _ in range(2000):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            size = np.random.randint(1, 4)
            
            # Get color from nearby pixel
            color = spatter[y, x]
            
            # Add random variation
            color_var = np.random.randint(-30, 30, 3)
            spatter_color = np.clip(color.astype(np.int16) + color_var, 0, 255)
            
            cv2.circle(spatter, (x, y), size, spatter_color.tolist(), -1)
        
        return spatter

    def sprayed_strokes(self, img_rgb, gray):
        """Photoshop Sprayed Strokes filter"""
        # Create airbrush-like strokes
        sprayed = cv2.bilateralFilter(img_rgb, 20, 100, 100)
        
        h, w = gray.shape
        
        # Add spray pattern
        for _ in range(5000):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            
            # Create small spray marks
            for _ in range(np.random.randint(3, 8)):
                dx = np.random.randint(-3, 3)
                dy = np.random.randint(-3, 3)
                
                if 0 <= x + dx < w and 0 <= y + dy < h:
                    # Blend with surrounding color
                    color = sprayed[y, x]
                    alpha = np.random.uniform(0.1, 0.3)
                    
                    sprayed[y + dy, x + dx] = (sprayed[y + dy, x + dx] * (1 - alpha) + 
                                            color * alpha).astype(np.uint8)
        
        return sprayed

    def sumi_e_filter(self, img_rgb, gray):
        """Photoshop Sumi-e (Japanese ink) filter"""
        # Convert to ink wash style
        sumi = np.ones_like(img_rgb) * 255
        
        # Create ink gradations
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Multiple ink density levels
        ink_levels = [200, 150, 100, 50, 20]
        
        for level in ink_levels:
            mask = blur < level
            ink_color = 255 - (255 - level) * 1.2
            ink_color = max(0, int(ink_color))
            sumi[mask] = [ink_color, ink_color, ink_color]
        
        # Add brush texture
        edges = cv2.Canny(gray, 30, 90)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7))
        brush_marks = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
        
        brush_mask = brush_marks == 255
        sumi[brush_mask] = [30, 30, 30]
        
        return sumi

    def underpainting(self, img_rgb, gray):
        """Photoshop Underpainting filter"""
        # Create underpainting effect
        under = cv2.bilateralFilter(img_rgb, 20, 100, 100)
        
        # Reduce saturation for underpainting look
        hsv = cv2.cvtColor(under, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 0.6)
        under = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add canvas texture
        h, w = gray.shape
        canvas_texture = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
        
        # Make texture follow image structure
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = edges == 255
        canvas_texture[edge_mask] *= 2
        
        under = np.clip(under.astype(np.int16) + canvas_texture, 0, 255).astype(np.uint8)
        
        return under

    def accented_edges(self, img_rgb, gray):
        """Photoshop Accented Edges filter"""
        # Enhance and accent edges
        edges = cv2.Canny(gray, 100, 200)
        
        # Create different edge treatments
        kernel = np.ones((3, 3), np.uint8)
        thick_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Base image with reduced contrast
        base = cv2.convertScaleAbs(img_rgb, alpha=0.7, beta=20)
        
        # Accent edges with bright color
        edge_mask = thick_edges == 255
        base[edge_mask] = [255, 255, 200]  # Bright yellow accent
        
        # Add secondary edges in different color
        thin_edges = cv2.Canny(gray, 50, 100)
        thin_mask = (thin_edges == 255) & (thick_edges == 0)
        base[thin_mask] = [200, 150, 255]  # Purple accent
        
        return base

    def bas_relief(self, img_rgb, gray):
        """Photoshop Bas Relief filter"""
        # Create 3D relief effect
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Create relief map
        relief = grad_x + grad_y
        relief = np.clip(relief + 128, 0, 255).astype(np.uint8)
        
        # Convert to color
        relief_color = cv2.cvtColor(relief, cv2.COLOR_GRAY2RGB)
        
        # Add original color information
        color_info = cv2.convertScaleAbs(img_rgb, alpha=0.3)
        
        result = cv2.addWeighted(relief_color, 0.7, color_info, 0.3, 0)
        
        return result

    def chalk_charcoal(self, img_rgb, gray):
        """Photoshop Chalk & Charcoal filter"""
        # Create chalk and charcoal effect
        h, w = gray.shape
        chalk_charcoal = np.ones((h, w, 3), dtype=np.uint8) * 128
        
        # Charcoal for dark areas
        dark_areas = gray < 100
        chalk_charcoal[dark_areas] = [30, 30, 30]
        
        # Chalk for light areas
        light_areas = gray > 180
        chalk_charcoal[light_areas] = [240, 240, 240]
        
        # Add texture
        texture = np.random.randint(-20, 20, (h, w, 3), dtype=np.int16)
        chalk_charcoal = np.clip(chalk_charcoal.astype(np.int16) + texture, 0, 255).astype(np.uint8)
        
        # Add some color hints
        color_reduced = cv2.bilateralFilter(img_rgb, 25, 100, 100)
        result = cv2.addWeighted(chalk_charcoal, 0.8, color_reduced, 0.2, 0)
        
        return result

    def conte_crayon_filter(self, img_rgb, gray):
        """Photoshop Conté Crayon filter"""
        # Create Conté crayon drawing effect
        conte = cv2.bilateralFilter(img_rgb, 15, 80, 80)
        
        # Reduce to earth tones
        hsv = cv2.cvtColor(conte, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = np.clip(hsv[:,:,0] + 10, 0, 179)  # Warm shift
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 0.7)  # Reduce saturation
        conte = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add crayon texture
        h, w = gray.shape
        crayon_texture = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
        
        # Make texture directional
        for i in range(h):
            for j in range(0, w, 4):
                end_j = min(j + 4, w)
                crayon_texture[i, j:end_j] = crayon_texture[i, j]
        
        conte = np.clip(conte.astype(np.int16) + crayon_texture, 0, 255).astype(np.uint8)
        
        return conte

    def graphic_pen(self, img_rgb, gray):
        """Photoshop Graphic Pen filter"""
        # Create graphic pen drawing
        edges = cv2.Canny(gray, 100, 200)
        
        # Create pen strokes
        h, w = gray.shape
        pen = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Add diagonal pen strokes based on intensity
        for i in range(0, h, 6):
            for j in range(0, w, 6):
                if i + 6 < h and j + 6 < w:
                    region = gray[i:i+6, j:j+6]
                    intensity = region.mean() / 255.0
                    
                    # Draw pen lines based on intensity
                    if intensity < 0.8:
                        cv2.line(pen, (j, i+6), (j+6, i), [0, 0, 0], 1)
                    if intensity < 0.6:
                        cv2.line(pen, (j, i+3), (j+6, i+3), [0, 0, 0], 1)
                    if intensity < 0.4:
                        cv2.line(pen, (j+3, i), (j+3, i+6), [0, 0, 0], 1)
                    if intensity < 0.2:
                        cv2.line(pen, (j, i), (j+6, i+6), [0, 0, 0], 1)
        
        # Add edge outlines
        edge_mask = edges == 255
        pen[edge_mask] = [0, 0, 0]
        
        return pen

    def halftone_pattern(self, img_rgb, gray):
        """Photoshop Halftone Pattern filter"""
        # Create halftone dot pattern
        h, w = gray.shape
        halftone = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        dot_size = 8
        for i in range(0, h, dot_size):
            for j in range(0, w, dot_size):
                if i + dot_size < h and j + dot_size < w:
                    # Get average intensity of region
                    region = gray[i:i+dot_size, j:j+dot_size]
                    intensity = region.mean() / 255.0
                    
                    # Calculate dot radius based on intensity
                    max_radius = dot_size // 2
                    radius = int(max_radius * (1 - intensity))
                    
                    if radius > 0:
                        center_x = j + dot_size // 2
                        center_y = i + dot_size // 2
                        cv2.circle(halftone, (center_x, center_y), radius, [0, 0, 0], -1)
        
        # Add color tinting
        color_avg = np.mean(img_rgb, axis=(0, 1))
        for c in range(3):
            halftone[:, :, c] = halftone[:, :, c] * (color_avg[c] / 255.0)
        
        return halftone.astype(np.uint8)

    def note_paper(self, img_rgb, gray):
        """Photoshop Note Paper filter"""
        # Create note paper texture
        h, w = gray.shape
        paper = np.ones((h, w, 3), dtype=np.uint8) * 240
        
        # Add paper texture
        paper_texture = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
        paper = np.clip(paper.astype(np.int16) + paper_texture, 230, 255).astype(np.uint8)
        
        # Add ruled lines
        for i in range(30, h, 40):
            cv2.line(paper, (20, i), (w-20, i), [180, 180, 255], 1)
        
        # Add margin line
        cv2.line(paper, (80, 0), (80, h), [255, 180, 180], 2)
        
        # Emboss effect from original image
        emboss = cv2.filter2D(gray, -1, np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]))
        emboss = np.clip(emboss + 128, 0, 255)
        
        # Apply emboss to paper
        emboss_3d = cv2.cvtColor(emboss, cv2.COLOR_GRAY2RGB)
        result = cv2.addWeighted(paper, 0.7, emboss_3d, 0.3, 0)
        
        return result

    def photocopy_effect(self, img_rgb, gray):
        """Photoshop Photocopy Effect filter"""
        # High contrast black and white
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        # Add photocopy artifacts
        h, w = gray.shape
        photocopy = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        # Add random specs and noise
        for _ in range(500):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            if np.random.random() > 0.5:
                cv2.circle(photocopy, (x, y), 1, [0, 0, 0], -1)
            else:
                cv2.circle(photocopy, (x, y), 1, [255, 255, 255], -1)
        
        # Add streaks
        for _ in range(50):
            x = np.random.randint(0, w)
            y1 = np.random.randint(0, h//2)
            y2 = np.random.randint(h//2, h)
            cv2.line(photocopy, (x, y1), (x, y2), [128, 128, 128], 1)
        
        return photocopy

    def plaster_effect(self, img_rgb, gray):
        """Photoshop Plaster Effect filter"""
        # Create plaster relief
        # Calculate gradients for 3D effect
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Create relief map
        plaster = grad_x * 0.5 + grad_y * 0.5 + 128
        plaster = np.clip(plaster, 0, 255).astype(np.uint8)
        
        # Convert to 3-channel
        plaster_rgb = cv2.cvtColor(plaster, cv2.COLOR_GRAY2RGB)
        
        # Add plaster texture
        h, w = gray.shape
        texture = np.random.randint(-20, 20, (h, w, 3), dtype=np.int16)
        plaster_rgb = np.clip(plaster_rgb.astype(np.int16) + texture, 0, 255).astype(np.uint8)
        
        # Tint with warm plaster color
        plaster_rgb[:, :, 0] = np.clip(plaster_rgb[:, :, 0] * 1.1, 0, 255)  # More red
        plaster_rgb[:, :, 1] = np.clip(plaster_rgb[:, :, 1] * 1.05, 0, 255)  # Slight green
        
        return plaster_rgb

    def reticulation(self, img_rgb, gray):
        """Photoshop Reticulation filter"""
        # Create reticulated (cracked) effect
        reticulated = cv2.bilateralFilter(img_rgb, 15, 80, 80)
        
        h, w = gray.shape
        
        # Create crack pattern
        cracks = np.zeros((h, w), dtype=np.uint8)
        
        # Add random crack lines
        for _ in range(200):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            length = np.random.randint(10, 50)
            angle = np.random.uniform(0, 2 * np.pi)
            
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            
            x2 = np.clip(x2, 0, w-1)
            y2 = np.clip(y2, 0, h-1)
            
            cv2.line(cracks, (x1, y1), (x2, y2), 255, 1)
        
        # Apply cracks to image
        crack_mask = cracks == 255
        reticulated[crack_mask] = reticulated[crack_mask] * 0.3
        
        # Add slight color separation
        reticulated[:, :, 0] = cv2.GaussianBlur(reticulated[:, :, 0], (3, 3), 0)
        reticulated[:, :, 2] = cv2.GaussianBlur(reticulated[:, :, 2], (5, 5), 0)
        
        return reticulated

    def stamp_filter(self, img_rgb, gray):
        """Photoshop Stamp filter"""
        # Create rubber stamp effect
        # High contrast threshold
        _, stamp = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        # Invert for stamp effect
        stamp = 255 - stamp
        
        # Add stamp texture
        h, w = gray.shape
        
        # Create uneven edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        stamp = cv2.morphologyEx(stamp, cv2.MORPH_CLOSE, kernel)
        stamp = cv2.morphologyEx(stamp, cv2.MORPH_OPEN, kernel)
        
        # Convert to color with stamp ink color
        stamp_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        stamp_mask = stamp == 255
        
        # Red ink color
        stamp_rgb[stamp_mask] = [180, 0, 0]
        stamp_rgb[~stamp_mask] = [255, 255, 255]
        
        # Add ink bleeding effect
        bleeding = cv2.GaussianBlur(stamp_rgb, (3, 3), 0)
        result = cv2.addWeighted(stamp_rgb, 0.8, bleeding, 0.2, 0)
        
        return result

    def water_paper(self, img_rgb, gray):
        """Photoshop Water Paper filter"""
        # Create wet paper effect
        water = cv2.bilateralFilter(img_rgb, 20, 100, 100)
        
        # Add water stains
        h, w = gray.shape
        
        # Create water spots
        for _ in range(100):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            radius = np.random.randint(10, 30)
            
            # Create water stain mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (x, y), radius, 255, -1)
            
            # Apply Gaussian blur to create water effect
            stain_mask = mask == 255
            water[stain_mask] = cv2.GaussianBlur(water, (15, 15), 0)[stain_mask]
            
            # Darken the edges of water spots
            edge_mask = cv2.Canny(mask, 50, 150) == 255
            water[edge_mask] = water[edge_mask] * 0.8
        
        # Add paper fiber texture
        fiber_texture = np.random.randint(-8, 8, (h, w, 3), dtype=np.int16)
        water = np.clip(water.astype(np.int16) + fiber_texture, 0, 255).astype(np.uint8)
        
        return water

    def clouds_filter(self, img_rgb, gray):
        """Photoshop Clouds filter"""
        h, w = gray.shape
        
        # Generate Perlin-like noise for clouds
        clouds = np.zeros((h, w), dtype=np.float32)
        
        # Multiple octaves of noise
        for octave in range(6):
            freq = 2 ** octave
            amplitude = 1.0 / freq
            
            noise = np.random.random((h // freq + 1, w // freq + 1))
            # Resize noise to full image size
            noise_resized = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
            clouds += noise_resized * amplitude
        
        # Normalize
        clouds = (clouds - clouds.min()) / (clouds.max() - clouds.min())
        clouds = (clouds * 255).astype(np.uint8)
        
        # Convert to RGB
        clouds_rgb = cv2.cvtColor(clouds, cv2.COLOR_GRAY2RGB)
        
        # Blend with original image
        result = cv2.addWeighted(img_rgb, 0.3, clouds_rgb, 0.7, 0)
        
        return result

    def difference_clouds(self, img_rgb, gray):
        """Photoshop Difference Clouds filter"""
        h, w = gray.shape
        
        # Generate two cloud layers
        clouds1 = np.zeros((h, w), dtype=np.float32)
        clouds2 = np.zeros((h, w), dtype=np.float32)
        
        # Generate noise for both layers
        for octave in range(5):
            freq = 2 ** octave
            amplitude = 1.0 / freq
            
            noise1 = np.random.random((h // freq + 1, w // freq + 1))
            noise2 = np.random.random((h // freq + 1, w // freq + 1))
            
            noise1_resized = cv2.resize(noise1, (w, h), interpolation=cv2.INTER_LINEAR)
            noise2_resized = cv2.resize(noise2, (w, h), interpolation=cv2.INTER_LINEAR)
            
            clouds1 += noise1_resized * amplitude
            clouds2 += noise2_resized * amplitude
        
        # Calculate difference
        diff_clouds = np.abs(clouds1 - clouds2)
        diff_clouds = (diff_clouds / diff_clouds.max() * 255).astype(np.uint8)
        
        # Convert to RGB and blend
        diff_clouds_rgb = cv2.cvtColor(diff_clouds, cv2.COLOR_GRAY2RGB)
        result = cv2.addWeighted(img_rgb, 0.4, diff_clouds_rgb, 0.6, 0)
        
        return result

    def fibers_filter(self, img_rgb, gray):
        """Photoshop Fibers filter"""
        h, w = gray.shape
        fibers = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Generate vertical fibers
        for _ in range(1000):
            x = np.random.randint(0, w)
            y_start = 0
            y_end = h
            
            # Random fiber color
            fiber_color = [
                np.random.randint(100, 255),
                np.random.randint(100, 255),
                np.random.randint(100, 255)
            ]
            
            # Draw wavy fiber
            points = []
            for y in range(y_start, y_end, 5):
                x_offset = int(np.sin(y * 0.1) * 3) + np.random.randint(-2, 2)
                points.append((x + x_offset, y))
            
            # Draw the fiber
            for i in range(len(points) - 1):
                cv2.line(fibers, points[i], points[i + 1], fiber_color, 1)
        
        # Blend with original
        result = cv2.addWeighted(img_rgb, 0.3, fibers, 0.7, 0)
        
        return result

    def lens_flare_filter(self, img_rgb, gray):
        """Photoshop Lens Flare filter"""
        h, w = gray.shape
        flare = img_rgb.copy()
        
        # Lens flare center
        center_x = w // 2
        center_y = h // 3
        
        # Main flare
        cv2.circle(flare, (center_x, center_y), 50, [255, 255, 200], -1)
        
        # Secondary flares
        for i, (scale, offset) in enumerate([(0.3, 0.5), (0.2, 0.8), (0.15, 1.2)]):
            flare_x = int(center_x + offset * (w // 4))
            flare_y = int(center_y + offset * (h // 6))
            radius = int(30 * scale)
            
            if 0 <= flare_x < w and 0 <= flare_y < h:
                cv2.circle(flare, (flare_x, flare_y), radius, [255, 200, 150], -1)
        
        # Add rays
        for angle in range(0, 360, 30):
            rad = np.radians(angle)
            end_x = int(center_x + 100 * np.cos(rad))
            end_y = int(center_y + 100 * np.sin(rad))
            cv2.line(flare, (center_x, center_y), (end_x, end_y), [255, 255, 150], 2)
        
        # Blend with original using screen blend mode
        flare_gray = cv2.cvtColor(flare, cv2.COLOR_RGB2GRAY)
        mask = flare_gray > 128
        
        result = img_rgb.copy()
        result[mask] = np.clip(result[mask].astype(np.float32) + flare[mask] * 0.5, 0, 255).astype(np.uint8)
        
        return result

    def lighting_effects(self, img_rgb, gray):
        """Photoshop Lighting Effects filter"""
        h, w = gray.shape
        lit = img_rgb.copy().astype(np.float32)
        
        # Create multiple light sources
        lights = [
            (w // 4, h // 4, 0.8, [255, 255, 200]),      # Warm top-left
            (3 * w // 4, h // 2, 0.6, [200, 200, 255]),  # Cool top-right
            (w // 2, 3 * h // 4, 0.4, [255, 200, 200])   # Pink bottom
        ]
        
        for light_x, light_y, intensity, color in lights:
            # Create distance map from light source
            y_coords, x_coords = np.ogrid[:h, :w]
            distances = np.sqrt((x_coords - light_x)**2 + (y_coords - light_y)**2)
            
            # Create light falloff
            max_dist = np.sqrt(w**2 + h**2) / 2
            light_map = np.clip(1.0 - distances / max_dist, 0, 1) * intensity
            
            # Apply colored lighting
            for c in range(3):
                lit[:, :, c] += light_map * color[c] * 0.3
        
        # Add ambient lighting
        lit *= 0.7  # Reduce overall brightness
        lit += 30   # Add ambient light
        
        return np.clip(lit, 0, 255).astype(np.uint8)

    def render_flames(self, img_rgb, gray):
        """Photoshop Render Flames filter"""
        h, w = gray.shape
        flames = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Generate flame effect
        for _ in range(500):
            # Start from bottom
            x = np.random.randint(0, w)
            y = h - 1
            
            flame_height = np.random.randint(h // 4, h // 2)
            
            # Draw flickering flame
            for i in range(flame_height):
                if y - i < 0:
                    break
                    
                # Flame color based on height
                heat = 1.0 - (i / flame_height)
                
                if heat > 0.8:
                    color = [255, int(255 * heat), 0]  # Yellow-white
                elif heat > 0.5:
                    color = [255, int(200 * heat), 0]  # Orange
                else:
                    color = [int(255 * heat), 0, 0]    # Red
                
                # Add flicker
                x_offset = int(np.sin(i * 0.5) * 5) + np.random.randint(-3, 3)
                flame_x = np.clip(x + x_offset, 0, w - 1)
                
                cv2.circle(flames, (flame_x, y - i), 2, color, -1)
        
        # Blur for flame effect
        flames = cv2.GaussianBlur(flames, (5, 5), 0)
        
        # Blend with original
        result = cv2.addWeighted(img_rgb, 0.4, flames, 0.6, 0)
        
        return result

    def tree_filter(self, img_rgb, gray):
        """Photoshop Tree filter"""
        h, w = gray.shape
        tree = np.zeros((h, w, 3), dtype=np.uint8)
        tree.fill(135)  # Sky color
        
        # Draw tree trunk
        trunk_x = w // 2
        trunk_bottom = h - 1
        trunk_top = h // 2
        trunk_width = 20
        
        cv2.rectangle(tree, 
                    (trunk_x - trunk_width//2, trunk_top), 
                    (trunk_x + trunk_width//2, trunk_bottom), 
                    [101, 67, 33], -1)  # Brown
        
        # Draw branches
        def draw_branch(x, y, angle, length, depth):
            if depth <= 0 or length < 5:
                return
                
            end_x = int(x + length * np.cos(angle))
            end_y = int(y - length * np.sin(angle))  # Negative for upward
            
            if 0 <= end_x < w and 0 <= end_y < h:
                cv2.line(tree, (int(x), int(y)), (end_x, end_y), [101, 67, 33], max(1, depth))
                
                # Recursive branches
                if depth > 1:
                    draw_branch(end_x, end_y, angle - 0.5, length * 0.7, depth - 1)
                    draw_branch(end_x, end_y, angle + 0.5, length * 0.7, depth - 1)
        
        # Draw main branches
        draw_branch(trunk_x, trunk_top, np.pi/2, 60, 4)
        draw_branch(trunk_x, trunk_top + 20, np.pi/2 - 0.3, 50, 3)
        draw_branch(trunk_x, trunk_top + 20, np.pi/2 + 0.3, 50, 3)
        
        # Add leaves
        for _ in range(200):
            x = np.random.randint(trunk_x - 100, trunk_x + 100)
            y = np.random.randint(trunk_top - 80, trunk_top + 40)
            
            if 0 <= x < w and 0 <= y < h:
                leaf_color = [34, 139, 34] if np.random.random() > 0.3 else [0, 100, 0]
                cv2.circle(tree, (x, y), np.random.randint(2, 5), leaf_color, -1)
        
        return tree

    def twirl_effect(self, img_rgb, gray):
        """Photoshop Twirl Effect filter"""
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        max_radius = min(w, h) // 2
        
        # Create coordinate maps
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Calculate distance and angle from center
        dx = x_coords - center_x
        dy = y_coords - center_y
        distances = np.sqrt(dx**2 + dy**2)
        angles = np.arctan2(dy, dx)
        
        # Apply twirl effect
        twirl_strength = 2.0
        twirl_angles = angles + (twirl_strength * (1 - distances / max_radius) * 
                                np.exp(-distances / (max_radius * 0.5)))
        
        # Calculate new coordinates
        new_x = center_x + distances * np.cos(twirl_angles)
        new_y = center_y + distances * np.sin(twirl_angles)
        
        # Clamp coordinates
        new_x = np.clip(new_x, 0, w - 1).astype(np.int32)
        new_y = np.clip(new_y, 0, h - 1).astype(np.int32)
        
        # Apply transformation
        twirled = np.zeros_like(img_rgb)
        twirled[y_coords, x_coords] = img_rgb[new_y, new_x]
        
        return twirled

    def wave_distortion(self, img_rgb, gray):
        """Photoshop Wave Distortion filter"""
        h, w = gray.shape
        
        # Create wave displacement maps
        y_coords, x_coords = np.mgrid[:h, :w]
        
        # Horizontal wave
        wave_x = 20 * np.sin(2 * np.pi * y_coords / 50)
        # Vertical wave
        wave_y = 15 * np.sin(2 * np.pi * x_coords / 40)
        
        # Apply displacement
        new_x = np.clip(x_coords + wave_x, 0, w - 1).astype(np.int32)
        new_y = np.clip(y_coords + wave_y, 0, h - 1).astype(np.int32)
        
        # Create result
        waved = np.zeros_like(img_rgb)
        waved[y_coords, x_coords] = img_rgb[new_y, new_x]
        
        return waved

    def zigzag_effect(self, img_rgb, gray):
        """Photoshop Zigzag Effect filter"""
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        
        # Create coordinate maps
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Calculate distance from center
        dx = x_coords - center_x
        dy = y_coords - center_y
        distances = np.sqrt(dx**2 + dy**2)
        
        # Apply zigzag ripple effect
        ripple_freq = 0.1
        ripple_amp = 10
        
        # Create ripple displacement
        ripple = ripple_amp * np.sin(distances * ripple_freq)
        
        # Apply displacement radially
        angles = np.arctan2(dy, dx)
        displacement_x = ripple * np.cos(angles)
        displacement_y = ripple * np.sin(angles)
        
        # Calculate new coordinates
        new_x = np.clip(x_coords + displacement_x, 0, w - 1).astype(np.int32)
        new_y = np.clip(y_coords + displacement_y, 0, h - 1).astype(np.int32)
        
        # Apply transformation
        zigzagged = np.zeros_like(img_rgb)
        zigzagged[y_coords, x_coords] = img_rgb[new_y, new_x]
        
        return zigzagged

# image_path = "C:/Users/Al Hikmah Computer/Desktop/PYTHON/pyautogui/safw.jpg"
def main():
    """Main function with examples and interface"""
    converter = EnhancedArtisticConverter()
    
    print("🎨 Enhanced Artistic Effects Converter - 130+ Styles")
    print("=" * 70)
    
    # Show all available effects
    converter.list_all_effects()
    
    # Example usage (replace with your image path)
    # image_path = "C:/Users/Al Hikmah Computer/Desktop/PYTHON/pyautogui/ras.jpg"  # Change this to your image path
    
    print(f"\n🚀 How to use this enhanced converter:")
    print(f"1. Replace the image path with your actual image location")
    print(f"2. Choose from 80 different artistic styles:")
    print(f"   • Traditional Art (1-19): Pencil, Oil, Watercolor, etc.")
    print(f"   • Digital Effects (20-35): Neon, Cyberpunk, Pop Art, etc.")
    print(f"   • Anime Styles (36-50): Shoujo, Kawaii, Cel Shading, etc.")
    print(f"   • Paint Styles (51-65): Fresco, Chinese Ink, Dutch Masters, etc.")
    print(f"   • Studio Ghibli (66-80): Totoro, Spirited Away, Forest Magic, etc.")
    print(f"\n📋 Available methods:")
    print(f"   - converter.apply_effect(image_path, effect_number)")
    print(f"   - create_comparison_grid(image_path, effects_list)")
    print(f"   - batch_convert_folder(input_folder, output_folder)")
    print(f"   - apply_preset_collection(image_path, collection_name)")
    
    try:
        # Showcase different categories
        print(f"\n🎭 Showcasing different style categories...")
        
        # Traditional art
        pencil = converter.apply_effect(image_path, 1)   # Pencil sketch
        
        # Digital effects
        neon = converter.apply_effect(image_path, 23)    # Neon glow
        
        # Anime styles
        kawaii = converter.apply_effect(image_path, 38)  # Kawaii style
        
        # Paint styles
        sumi_e = converter.apply_effect(image_path, 56)  # Japanese Sumi-e
        
        # Ghibli styles
        totoro = converter.apply_effect(image_path, 69)  # Totoro style
        
        print(f"\n✨ Creating comparison grids...")
        
        # Create themed comparison grids
        create_comparison_grid(image_path, [1, 4, 5, 23, 29])  # Mixed classics
        create_comparison_grid(image_path, [36, 38, 40, 42, 47])  # Anime showcase
        create_comparison_grid(image_path, [66, 68, 69, 77, 80])  # Ghibli showcase
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure:")
        print("  • Your image path is correct")  
        print("  • Required libraries are installed:")
        print("    pip install opencv-python matplotlib scipy numpy")
        print("  • Image file is accessible and in supported format")

def batch_convert_folder(input_folder, output_folder, effects_list=None):
    """Convert all images in a folder with specified effects"""
    import os
    
    converter = EnhancedArtisticConverter()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Enhanced default selection covering all categories
    if effects_list is None:
        effects_list = [1, 4, 23, 29, 38, 56, 69, 77]  # Representative from each category
    
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    print(f"🔄 Batch processing folder: {input_folder}")
    print(f"📁 Output folder: {output_folder}")
    print(f"🎨 Effects to apply: {effects_list}")
    
    processed_count = 0
    
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(fmt) for fmt in supported_formats):
            input_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            
            print(f"\n📸 Processing: {filename}")
            
            for effect_num in effects_list:
                try:
                    if effect_num in converter.effects:
                        effect_name = converter.effects[effect_num][0]
                        result = converter.apply_effect(input_path, effect_num)
                        
                        # Save result
                        safe_name = effect_name.replace(' ', '_').replace('/', '_').lower()
                        output_filename = f"{base_name}_{safe_name}.jpg"
                        output_path = os.path.join(output_folder, output_filename)
                        
                        if len(result.shape) == 3:
                            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(output_path, result_bgr)
                        else:
                            cv2.imwrite(output_path, result)
                        
                        print(f"  ✅ {effect_name} -> {output_filename}")
                    else:
                        print(f"  ⚠️  Effect {effect_num} not found")
                        
                except Exception as e:
                    print(f"  ❌ Error with effect {effect_num}: {e}")
            
            processed_count += 1
    
    print(f"\n🎉 Batch processing complete! Processed {processed_count} images.")

def create_comparison_grid(image_path, effects_selection):
    """Create a comparison grid with selected effects"""
    converter = EnhancedArtisticConverter()
    img_rgb, gray = converter.load_and_preprocess(image_path)
    
    num_effects = len(effects_selection)
    cols = min(4, num_effects + 1)  # +1 for original
    rows = (num_effects + 1 + cols - 1) // cols  # Ceiling division
    
    plt.figure(figsize=(cols * 4, rows * 4))
    
    # Original image
    plt.subplot(rows, cols, 1)
    plt.imshow(img_rgb)
    plt.title('🖼️  Original', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Apply selected effects
    for i, effect_num in enumerate(effects_selection, 2):
        if effect_num in converter.effects:
            effect_name, effect_func = converter.effects[effect_num]
            
            try:
                result = effect_func(img_rgb, gray)
                
                plt.subplot(rows, cols, i)
                if len(result.shape) == 3:
                    plt.imshow(result)
                else:
                    plt.imshow(result, cmap='gray')
                plt.title(f'{effect_num}. {effect_name}', fontsize=10)
                plt.axis('off')
                
            except Exception as e:
                plt.subplot(rows, cols, i)
                plt.text(0.5, 0.5, f'❌ Error:\n{effect_name}\n{str(e)[:30]}...', 
                        ha='center', va='center', transform=plt.gca().transAxes,
                        fontsize=8, color='red')
                plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Enhanced preset collections for the new 80-effect system
ENHANCED_PRESET_COLLECTIONS = {
    'traditional_sketching': [1, 2, 3, 8, 9, 14, 15],  # Traditional sketching
    'classic_painting': [4, 5, 6, 7, 51, 52, 63, 64],  # Classic painting styles
    'modern_digital': [23, 24, 27, 28, 30, 34],        # Modern/digital art
    'anime_showcase': [36, 38, 40, 42, 47, 49],        # Anime styles
    'eastern_art': [55, 56],                           # Chinese and Japanese art
    'ghibli_magic': [66, 68, 69, 77, 78, 80],         # Studio Ghibli styles
    'artistic_masters': [57, 58, 59, 63, 64, 65],     # Art history masters
    'special_effects': [20, 21, 22, 26],              # Special vision effects
    'vintage_classic': [16, 17, 18, 19, 25],          # Vintage techniques
    'kawaii_cute': [38, 39, 44, 47],                  # Kawaii and cute styles
    'dark_dramatic': [41, 46, 70],                    # Dark and dramatic
    'dreamy_soft': [74, 79, 80],                      # Soft and dreamy
    'nature_landscape': [66, 76, 77],                 # Nature and landscapes
    'pop_culture': [27, 28, 29, 48]                   # Pop culture styles
}

def apply_preset_collection(image_path, collection_name='traditional_sketching'):
    """Apply a preset collection of effects with enhanced options"""
    if collection_name not in ENHANCED_PRESET_COLLECTIONS:
        print(f"🎨 Available collections:")
        for name, effects in ENHANCED_PRESET_COLLECTIONS.items():
            converter = EnhancedArtisticConverter()
            effect_names = [converter.effects[e][0] for e in effects if e in converter.effects]
            print(f"  📂 {name}: {effect_names}")
        return
    
    effects = ENHANCED_PRESET_COLLECTIONS[collection_name]
    print(f"🎭 Applying '{collection_name}' collection")
    print(f"🎨 Effects: {effects}")
    
    create_comparison_grid(image_path, effects)

def create_style_showcase(image_path):
    """Create a comprehensive showcase of different style categories"""
    print("🌟 Creating comprehensive style showcase...")
    
    # Showcase each major category
    categories = {
        'Traditional Art': [1, 4, 5, 8],
        'Anime Styles': [36, 38, 40, 47], 
        'Paint Masters': [55, 56, 57, 63],
        'Ghibli Magic': [68, 69, 77, 80],
        'Digital Effects': [23, 24, 27, 30]
    }
    
    for category_name, effects in categories.items():
        print(f"\n🎨 {category_name} Showcase:")
        create_comparison_grid(image_path, effects)

def interactive_effect_selector(image_path):
    """Interactive effect selection interface"""
    converter = EnhancedArtisticConverter()
    
    print("🎮 Interactive Effect Selector")
    print("=" * 50)
    
    while True:
        print(f"\n📋 Options:")
        print(f"1. View all effects")
        print(f"2. Apply single effect")
        print(f"3. Apply preset collection")
        print(f"4. Create custom comparison")
        print(f"5. Style showcase")
        print(f"0. Exit")
        
        choice = input(f"\n🔹 Enter your choice (0-5): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            converter.list_all_effects()
        elif choice == '2':
            effect_num = int(input("Enter effect number (1-150): "))
            if 1 <= effect_num <= 150:
                converter.apply_effect(image_path, effect_num)
            else:
                print("❌ Invalid effect number!")
        elif choice == '3':
            collection = input("Enter collection name: ").strip()
            apply_preset_collection(image_path, collection)
        elif choice == '4':
            effects_str = input("Enter effect numbers (comma-separated): ")
            effects = [int(x.strip()) for x in effects_str.split(',')]
            create_comparison_grid(image_path, effects)
        elif choice == '5':
            create_style_showcase(image_path)
        else:
            print("❌ Invalid choice!")

def simple_effect_generator():
    """Simple function to take user input and generate effects"""
    converter = EnhancedArtisticConverter()
    
    # Set your image path here
    # image_path = "C:/Users/Al Hikmah Computer/Desktop/PYTHON/pyautogui/ras.jpg"
    
    print("🎨 Simple Effect Generator")
    print("=" * 40)
    print("Available effects: 1-80")
    print("Categories:")
    print("  1-19: Traditional Art")
    print("  20-35: Digital Effects") 
    print("  36-50: Anime Styles")
    print("  51-65: Paint Styles")
    print("  66-80: Studio Ghibli")
    print("=" * 40)
    
    while True:
        try:
            # Get user input
            user_input = input("\n📝 Enter effect number (1-80) or 'quit' to exit: ").strip().lower()
            
            if user_input == 'quit' or user_input == 'q':
                print("👋 Goodbye!")
                break
            
            # Convert to integer
            effect_number = int(user_input)
            
            # Validate range
            if 1 <= effect_number <= 300:
                print(f"🎭 Applying effect {effect_number}...")
                
                # Get effect name
                effect_name = converter.effects[effect_number][0]
                print(f"✨ Effect: {effect_name}")
                
                # Apply effect
                result = converter.apply_effect(image_path, effect_number)
                print(f"✅ {effect_name} applied successfully!")
                
            else:
                print("❌ Please enter a number between 1 and 80")
                
        except ValueError:
            print("❌ Please enter a valid number or 'quit'")
        except FileNotFoundError:
            print("❌ Image file not found! Please check the image path.")
            print(f"Current path: {image_path}")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def quick_effect_demo():
    """Quick demo function - just enter number and see result"""
    converter = EnhancedArtisticConverter()
    # image_path = "C:/Users/Al Hikmah Computer/Desktop/PYTHON/pyautogui/ras.jpg"
    
    print("🚀 Quick Effect Demo")
    print("Just enter any number 1-80 to see the effect!")
    
    # Show some popular effects as examples
    popular_effects = {
        1: "Pencil Sketch",
        4: "Oil Painting", 
        23: "Neon Glow",
        29: "Anime Style",
        38: "Kawaii Style",
        56: "Japanese Sumi-e",
        69: "My Neighbor Totoro",
        77: "Ghibli Forest"
    }
    
    print("\n🌟 Popular effects to try:")
    for num, name in popular_effects.items():
        print(f"  {num}: {name}")
    
    while True:
        try:
            effect_num = input(f"\n🎯 Enter effect number (1-150): ").strip()
            
            if effect_num.lower() in ['quit', 'q', 'exit']:
                break
                
            effect_num = int(effect_num)
            
            if 1 <= effect_num <= 200:
                # Apply effect immediately
                converter.apply_effect(image_path, effect_num)
            else:
                print("❌ Number must be between 1-150")
                
        except ValueError:
            print("❌ Please enter a valid number")
        except Exception as e:
            print(f"❌ Error: {e}")
            break
def apply_all_effects_grid(image_path, output_folder="effect_grids", effects_per_grid=15):
    """
    Apply all effects to image and save as grids with 15 effects per image
    """
    import os
    import matplotlib.pyplot as plt
    
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    converter = EnhancedArtisticConverter()
    img_rgb, gray = converter.load_and_preprocess(image_path)
    
    # Get all effect numbers
    all_effects = list(converter.effects.keys())
    total_effects = len(all_effects)
    
    print(f"🎨 Applying {total_effects} effects in grids of {effects_per_grid}")
    
    # Create grids
    grid_count = 0
    for i in range(0, total_effects, effects_per_grid):
        grid_count += 1
        effects_batch = all_effects[i:i + effects_per_grid]
        
        # Calculate grid dimensions (4x4 grid = 16 slots, use 15 for effects + 1 original)
        cols = 4
        rows = 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
        fig.suptitle(f'Effects Grid {grid_count} - Effects {i+1} to {min(i+effects_per_grid, total_effects)}', 
                     fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # First slot: Original image
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original', fontsize=10, fontweight='bold')
        axes[0].axis('off')
        
        # Apply effects
        for idx, effect_num in enumerate(effects_batch, 1):
            if idx < 16:  # Max 15 effects + 1 original = 16 slots
                effect_name, effect_func = converter.effects[effect_num]
                
                try:
                    # Apply effect
                    result = effect_func(img_rgb, gray)
                    
                    # Display
                    axes[idx].imshow(result, cmap='gray' if len(result.shape) == 2 else None)
                    axes[idx].set_title(f'{effect_num}. {effect_name}', fontsize=8)
                    axes[idx].axis('off')
                    
                    print(f"  ✅ Applied: {effect_name}")
                    
                except Exception as e:
                    # Show error in grid
                    axes[idx].text(0.5, 0.5, f'Error:\n{effect_name}', 
                                  ha='center', va='center', fontsize=8, color='red')
                    axes[idx].axis('off')
                    print(f"  ❌ Error: {effect_name}")
        
        # Hide unused slots
        for idx in range(len(effects_batch) + 1, 16):
            axes[idx].axis('off')
        
        # Save grid
        plt.tight_layout()
        grid_filename = f"{output_folder}/effects_grid_{grid_count:02d}.jpg"
        plt.savefig(grid_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved: {grid_filename}")
    
    print(f"\n🎉 Complete! Created {grid_count} grids in '{output_folder}' folder")
    return grid_count

# Usage:
# apply_all_effects_grid(image_path, "my_effects", 15)

def batch_effect_generator():
    """Generate multiple effects from user input"""
    converter = EnhancedArtisticConverter()
    # image_path = "C:/Users/Al Hikmah Computer/Desktop/PYTHON/pyautogui/ras.jpg"
    
    print("🎨 Batch Effect Generator")
    print("Enter multiple effect numbers separated by commas")
    print("Example: 1,4,23,38,69")
    
    try:
        user_input = input("\n📝 Enter effect numbers: ").strip()
        
        # Parse numbers
        effect_numbers = [int(x.strip()) for x in user_input.split(',')]
        
        # Validate all numbers
        valid_numbers = [num for num in effect_numbers if 1 <= num <= 150]
        
        if not valid_numbers:
            print("❌ No valid effect numbers found!")
            return
            
        print(f"🎭 Applying {len(valid_numbers)} effects...")
        
        # Create comparison grid
        create_comparison_grid(image_path, valid_numbers)
        
    except ValueError:
        print("❌ Please enter valid numbers separated by commas")
    except Exception as e:
        print(f"❌ Error: {e}")

def run_effect_generator():
    """Main function to run the effect generator"""
    print("🎨 Artistic Effect Generator")
    print("=" * 50)
    print("Choose mode:")
    print("1. Single Effect Mode (enter one number at a time)")
    print("2. Quick Demo Mode (popular effects)")  
    print("3. Batch Mode (multiple effects at once)")
    print("4. Interactive Menu")
    
    try:
        mode = input("\nSelect mode (1-4): ").strip()
        
        # image_path = "C:/Users/Al Hikmah Computer/Desktop/PYTHON/pyautogui/ras.jpg"
        
        if mode == '1':
            simple_effect_generator()
        elif mode == '2':
            quick_effect_demo()
        elif mode == '3':
            batch_effect_generator()
        elif mode == '4':
            interactive_effect_selector(image_path)
        else:
            print("❌ Invalid mode. Running single effect mode...")
            simple_effect_generator()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

def apply_effect_now(effect_number):
    """One-liner to apply effect immediately"""
    converter = EnhancedArtisticConverter()
    # image_path = "C:/Users/Al Hikmah Computer/Desktop/PYTHON/pyautogui/ras.jpg"
    
    if 1 <= effect_number <= 150:
        return converter.apply_effect(image_path, effect_number)
    else:
        print("❌ Effect number must be between 1-80")
        return None

def show_random_effects(count=5):
    """Show random effects for inspiration"""
    import random
    converter = EnhancedArtisticConverter()
    # image_path = "C:/Users/Al Hikmah Computer/Desktop/PYTHON/pyautogui/ras.jpg"
    
    print(f"🎲 Showing {count} random effects...")
    
    # Get random effect numbers
    random_effects = random.sample(range(1, 81), count)
    
    print(f"🎨 Random effects: {random_effects}")
    
    # Show them in a comparison grid
    create_comparison_grid(image_path, random_effects)

def list_effects_by_category():
    """List all effects organized by category"""
    converter = EnhancedArtisticConverter()
    
    categories = {
        "🎨 Traditional Art (1-19)": range(1, 20),
        "💻 Digital Effects (20-35)": range(20, 36),
        "🎌 Anime Styles (36-50)": range(36, 51),
        "🖌️ Paint Styles (51-65)": range(51, 66),
        "🏰 Studio Ghibli (66-80)": range(66, 81)
    }
    
    print("📋 All Available Effects by Category")
    print("=" * 60)
    
    for category, effect_range in categories.items():
        print(f"\n{category}:")
        for i in effect_range:
            if i in converter.effects:
                name, _ = converter.effects[i]
                print(f"  {i:2d}. {name}")

if __name__ == "__main__":

   # main()
    
    run_effect_generator()
    
    
    # simple_effect_generator()
    
  
    # quick_effect_demo()
    
    # Option 5: Apply effect directly
    # apply_effect_now(23)  # Neon glow
    # apply_effect_now(69)  # Totoro style
    
    # Option 6: Show random effects for inspiration
    # show_random_effects(5)
    
    # Option 7: List all effects by category
    # list_effects_by_category()
    # Apply all effects and create grids with 15 effects each
    # apply_all_effects_grid(image_path, "effect_grids", 15)