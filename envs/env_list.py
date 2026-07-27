"""Task index constants and name mappings for the Meta-World MT10 toy setup."""

# --- Environment index constants ---
REACH_IDX                = 0   # SawyerReachEnvV3
PUSH_IDX                 = 1   # SawyerPushEnvV3
PICK_PLACE_IDX           = 2   # SawyerPickPlaceEnvV3
DOOR_OPEN_IDX            = 3   # SawyerDoorEnvV3
DRAWER_OPEN_IDX          = 4   # SawyerDrawerOpenEnvV3
DRAWER_CLOSE_IDX         = 5   # SawyerDrawerCloseEnvV3
BUTTON_PRESS_TOPDOWN_IDX = 6   # SawyerButtonPressTopdownEnvV3
PEG_INSERTION_SIDE_IDX   = 7   # SawyerPegInsertionSideEnvV3
WINDOW_OPEN_IDX          = 8   # SawyerWindowOpenEnvV3
WINDOW_CLOSE_IDX         = 9   # SawyerWindowCloseEnvV3

# --- Index -> Meta-World task id (used to create the env) ---
IDX_TO_ENV_NAME = {
    REACH_IDX:                "reach-v3",
    PUSH_IDX:                 "push-v3",
    PICK_PLACE_IDX:           "pick-place-v3",
    DOOR_OPEN_IDX:            "door-open-v3",
    DRAWER_OPEN_IDX:          "drawer-open-v3",
    DRAWER_CLOSE_IDX:         "drawer-close-v3",
    BUTTON_PRESS_TOPDOWN_IDX: "button-press-topdown-v3",
    PEG_INSERTION_SIDE_IDX:   "peg-insert-side-v3",
    WINDOW_OPEN_IDX:          "window-open-v3",
    WINDOW_CLOSE_IDX:         "window-close-v3",
}

# --- Index -> natural-language instruction (for the VLA text encoder) ---
IDX_TO_INSTRUCTION = {
    REACH_IDX:                "reach the target position",
    PUSH_IDX:                 "push the puck to the goal",
    PICK_PLACE_IDX:           "pick up the object and place it at the target",
    DOOR_OPEN_IDX:            "open the door",
    DRAWER_OPEN_IDX:          "open the drawer",
    DRAWER_CLOSE_IDX:         "close the drawer",
    BUTTON_PRESS_TOPDOWN_IDX: "press the button from above",
    PEG_INSERTION_SIDE_IDX:   "insert the peg into the side hole",
    WINDOW_OPEN_IDX:          "open the window",
    WINDOW_CLOSE_IDX:         "close the window",
}

# --- The task you're currently working on ---
CHOSEN_IDX = WINDOW_CLOSE_IDX
CHOSEN_ENV_NAME = IDX_TO_ENV_NAME[CHOSEN_IDX]        # "drawer-open-v3"
CHOSEN_INSTRUCTION = IDX_TO_INSTRUCTION[CHOSEN_IDX]  # "open the drawer"
