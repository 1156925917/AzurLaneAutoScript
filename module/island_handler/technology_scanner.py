import cv2
import numpy as np
from yaml import safe_dump

from module.base.button import ButtonGrid
from module.base.decorator import cached_property
from module.base.mask import Mask
from module.base.utils import color_similarity_2d, rgb2luma, load_image, random_rectangle_vector, area_offset, crop
from module.island_handler.assets import *
from module.island.data import DIC_ISLAND_TECHNOLOGY
from module.island.ui import IslandUI
from module.logger import logger
from module.ui.navbar import Navbar
from module.ui.page import page_island_technology

DELTA_X = 136 + 2/3
DELTA_Y = 60
ORIGIN_X = -5/3
ORIGIN_Y = 46
LEFT_STRIP = 167
MASK_ISLAND_TECHNOLOGY = Mask('./assets/mask/MASK_ISLAND_TECHNOLOGY.png')
TECHNOLOGY_LENGTH = {
    '2': 3139 - 1280 + LEFT_STRIP,
    '3': 4231 - 1280 + LEFT_STRIP,
    '4': 3003 - 1280 + LEFT_STRIP,
    '5': 5462 - 1280 + LEFT_STRIP,
    '6': 4233 - 1280 + LEFT_STRIP,
}
DETECTION_AREA = (167, 54, 1280, 720)
DETECTION_AREA_MASK = (1098, 646, 1280, 720)
BUTTON_AREA = (-110, -26, 110, 26)
MIN_VIEW_MATCH_SIMILARITY = 0.08
SCAN_MAX_STEPS = {
    2: 8,
    3: 10,
    4: 8,
    5: 14,
    6: 10,
}


def extract_flowchart(image):
    brightness = rgb2luma(image)
    black = color_similarity_2d(image, (7, 10, 17))
    brightness_mask = cv2.inRange(brightness, 160, 255)
    black_mask = cv2.inRange(black, 245, 255)
    mask = cv2.bitwise_or(brightness_mask, black_mask)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    filled_mask = MASK_ISLAND_TECHNOLOGY.apply(filled_mask)
    filled_mask = filled_mask[:, LEFT_STRIP:]
    return filled_mask

def get_technology_tab_and_position(index):
    tech_info = DIC_ISLAND_TECHNOLOGY[index]
    tab = tech_info['tech_belong']
    axis_x, axis_y = tech_info['axis']
    position_x = ORIGIN_X + DELTA_X * axis_x
    position_y = ORIGIN_Y + DELTA_Y * axis_y
    return tab, (position_x, position_y)


class IslandTechnologyScanner(IslandUI):
    """
    Currently only supports checking tab 2,3,4,5,6.
    """
    @cached_property
    def _island_technology_side_navbar(self):
        island_technology_side_navbar = ButtonGrid(
            origin=(13, 107), delta=(0, 196/3),
            button_shape=(128, 43), grid_shape=(1, 5)
        )
        return Navbar(grids=island_technology_side_navbar,
                      active_color=(30, 143, 255),
                      inactive_color=(50, 52, 55),
                      active_count=500,
                      inactive_count=500)

    def _island_technology_side_navbar_get_active(self):
        active, _, _ = self._island_technology_side_navbar.get_info(main=self)
        if active is None:
            return 1
        return active + 2

    def island_technology_side_navbar_ensure(self, tab=1, skip_first_screenshot=True):
        """
        Tab 2, 3, 4, 5, 6 corresponds to _island_technology_side_navbar 1, 2, 3, 4, 5
        Tab 1 is a special situation where the botton icon is chosen,
        and all the navbar icons are inactive.
        """
        logger.info(f'Ensure island technology tab {tab}')
        for _ in self.loop(skip_first=skip_first_screenshot, timeout=15):
            active = self._island_technology_side_navbar_get_active()
            if active == tab:
                return True
            if tab == 1:
                self.device.click(ISLAND_TECHNOLOGY_TAB1)
                continue
            else:
                if active == 1:
                    self.device.click(self._island_technology_side_navbar.grids.buttons[tab-2])
                    continue
                else:
                    return self._island_technology_side_navbar.set(self, upper=tab-1)
        logger.warning(f'Failed to ensure island technology tab {tab}')
        return False

    def get_technology_view_position(self, tab, detail=False):
        globe_view = load_image(f'./assets/island/technology/technology_chart_{tab}.png')
        extracted_flowchart = extract_flowchart(self.device.image)
        result = cv2.matchTemplate(globe_view, extracted_flowchart, cv2.TM_CCOEFF_NORMED)
        _, similarity, _, loca = cv2.minMaxLoc(result)
        if similarity < MIN_VIEW_MATCH_SIMILARITY:
            logger.warning(
                f'Island technology tab {tab} view match is weak: '
                f'position={loca[0]}, similarity={similarity:.3f}'
            )
        if detail:
            return loca[0], similarity
        return loca[0]

    def _island_technology_swipe(self, forward=True):
        detection_area = DETECTION_AREA
        direction_vector = (-600, 0) if forward else (600, 0)
        p1, p2 = random_rectangle_vector(
            direction_vector, box=detection_area, random_range=(-50, -50, 50, 50), padding=20
        )
        self.device.drag(p1, p2, segments=2, shake=(0, 25), point_random=(0, 0, 0, 0), shake_random=(0, -5, 0, 5))

    def technology_reset_view(self, skip_first_screenshot=True):
        active = self._island_technology_side_navbar_get_active()
        if active not in SCAN_MAX_STEPS:
            logger.warning(f'Unexpected island technology active tab {active}, skip reset')
            return False
        logger.info(f'Reset island technology tab {active} view')
        for _ in range(10):  # tab 5 has 4400 length, so 5 swipes are not enough
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()
            position_x, similarity = self.get_technology_view_position(tab=active, detail=True)
            logger.info(f'Island technology tab {active} reset position={position_x}, similarity={similarity:.3f}')
            if position_x < 3:
                self.device.click_record_remove('DRAG')
                return True
            if similarity < MIN_VIEW_MATCH_SIMILARITY:
                self.device.screenshot()
            self._island_technology_swipe(forward=False)
        self.device.click_record_remove('DRAG')
        return False

    def scan_all(self):
        all_technology = {}
        for index in DIC_ISLAND_TECHNOLOGY.keys():
            if DIC_ISLAND_TECHNOLOGY[index]['tech_belong'] not in [2, 3, 4, 5, 6]:
                continue
            tab, position = get_technology_tab_and_position(index)
            all_technology[index] = {
                'tab': tab,
                'position': position,
                'active': False,
            }
        technology_by_tab = [{} for _ in range(5)]
        for index, info in all_technology.items():
            technology_by_tab[info['tab'] - 2][index] = info['position']
        for tab in [2, 3, 4, 5, 6]:
            logger.hr(f'Scan island technology tab {tab}', level=2)
            if not self.island_technology_side_navbar_ensure(tab=tab):
                continue
            self.technology_reset_view()
            position_x_old = None
            matched_count_before = sum(1 for info in all_technology.values() if info['active'])
            for step, _ in enumerate(self.loop()):
                if step >= SCAN_MAX_STEPS[tab]:
                    logger.warning(f'Island technology tab {tab} reached scan step limit')
                    break
                position_x, similarity = self.get_technology_view_position(tab=tab, detail=True)
                logger.info(f'Island technology tab {tab} scan step={step}, position={position_x}, similarity={similarity:.3f}')
                if position_x_old is not None:
                    if position_x - position_x_old < 5:
                        break
                position_x_old = position_x
                for index, (tech_pos_x, tech_pos_y) in technology_by_tab[tab - 2].items():
                    tech_pos_x_in_view = tech_pos_x - position_x
                    if (DETECTION_AREA[0] - BUTTON_AREA[0] <= LEFT_STRIP + tech_pos_x_in_view <= DETECTION_AREA[2] - BUTTON_AREA[2]
                        and not (
                            tech_pos_y > DETECTION_AREA_MASK[1] + BUTTON_AREA[1]
                            and LEFT_STRIP + tech_pos_x_in_view >= DETECTION_AREA_MASK[0]
                            )):
                        tech_button = crop(self.device.image, area=area_offset(BUTTON_AREA, (LEFT_STRIP + tech_pos_x_in_view, tech_pos_y)))
                        luma = rgb2luma(tech_button)
                        color = np.mean(luma.flatten())
                        bright_ratio = np.count_nonzero(luma > 160) / luma.size
                        if color > 150 or bright_ratio > 0.12:
                            all_technology[index]['active'] = True
                self._island_technology_swipe(forward=True)
                self.device.click_record_remove('DRAG')
            matched_count_after = sum(1 for info in all_technology.values() if info['active'])
            logger.info(f'Island technology tab {tab} detected {matched_count_after - matched_count_before} active technologies')
        return {index: info['active'] for index, info in all_technology.items()}

    def get_technology_status(self, dump_key=None):
        logger.hr('Scan island technology')
        self.ui_ensure(page_island_technology)
        result = self.scan_all()
        if dump_key is not None:
            value = safe_dump(result)
            self.config.cross_set(keys=dump_key, value=value)
        return result
