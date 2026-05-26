from module.base.button import ButtonGrid
from module.island.ui import IslandUI
from module.island_handler.assets import *
from module.ui.switch import Switch

ISLAND_DOCK_SORTING = Switch('island_dock_sorting')
ISLAND_DOCK_SORTING.add_state('Ascending', check_button=ISLAND_DOCK_SORT_ASC, 
                              click_button=ISLAND_DOCK_SORTING_CLICK)
ISLAND_DOCK_SORTING.add_state('Descending', check_button=ISLAND_DOCK_SORT_DESC, 
                              click_button=ISLAND_DOCK_SORTING_CLICK)

ISLAND_DOCK_CARD_GRIDS = ButtonGrid(
    origin=(56, 139), delta=(140, 180), button_shape=(124, 164), grid_shape=(6, 2), name='CARD'
)

class IslandDock(IslandUI):
    def is_in_island_dock(self):
        return self.appear(ISLAND_DOCK_CHECK, offset=(20, 20))

    def handle_island_dock_loading(self):
        for _ in self.loop(timeout=1.2):
            pass

    def _island_dock_quit_check_func(self):
        return not self.appear(ISLAND_DOCK_CHECK, offset=(20, 20))

    def island_dock_quit(self):
        self.ui_back(check_button=self._island_dock_quit_check_func, skip_first_screenshot=True)

    def island_dock_sort_method_dsc_set(self, enable=True, wait_loading=True):
        """
        Args:
            enable (bool): True to set descending sorting
            wait_loading (bool): Default to True, use False on continuous operation 
        """
        if ISLAND_DOCK_SORTING.set('Descending' if enable else 'Ascending', main=self):
            if wait_loading:
                self.handle_island_dock_loading()
            return True
        return False

    def island_dock_select_one(self, button, skip_first=True):
        """
        Args:
            button (Button): Character button to select
            skip_first (bool):
        """
        self.interval_clear(ISLAND_DOCK_CHECK)
        for _ in self.loop(skip_first=skip_first):
            if self.is_button_selected(button, color=(19, 181, 231)):
                break

            if self.appear(ISLAND_DOCK_CHECK, offset=(20, 20), interval=5):
                self.device.click(button)
                continue

    def island_dock_select_confirm(self, check_button, skip_first=True):
        """
        Args:
            check_button (callable, Button):
            skip_first (bool):
        """
        for _ in self.loop(skip_first=skip_first):
            if self.ui_process_check_button(check_button):
                break

            if self.appear_then_click(ISLAND_DOCK_CHARACTER_CONFIRM, offset=(20, 20), interval=5):
                continue


