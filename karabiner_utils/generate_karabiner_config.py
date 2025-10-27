import json


def _generate_karabiner_simlayer_manipulators(
    layer_name: str, layer_key: str, from_key: str, to_action: dict
) -> list:
    """Generate a 'simlayer' config for Karabiner-Elements.

    In my opinion, simlayers provide the most natural layer experience in Karabiner-Elements. But, they are painful to
    write by hand.
    """
    return [
        {
            "conditions": [{"name": layer_name, "type": "variable_if", "value": 1}],
            "from": {"key_code": from_key, "modifiers": {"optional": ["any"]}},
            "to": [to_action],
            "type": "basic",
        },
        {
            "from": {
                "modifiers": {"optional": ["any"]},
                "simultaneous": [{"key_code": layer_key}, {"key_code": from_key}],
                "simultaneous_options": {
                    "detect_key_down_uninterruptedly": True,
                    "key_down_order": "strict",
                    "key_up_order": "strict_inverse",
                    "to_after_key_up": [
                        {"set_variable": {"name": layer_name, "value": 0}}
                    ],
                },
            },
            "to": [
                {"set_variable": {"name": layer_name, "value": 1}},
                to_action,
            ],
            "type": "basic",
        },
    ]


def main():
    layer_name = "nav_layer"
    layer_key = "spacebar"

    mappings = {
        # VIM navigation keys
        "h": {"key_code": "left_arrow", "modifiers": []},
        "j": {"key_code": "down_arrow", "modifiers": []},
        "k": {"key_code": "up_arrow", "modifiers": []},
        "l": {"key_code": "right_arrow", "modifiers": []},
        # Application shortcuts
        "a": {"shell_command": "open -a 'Visual Studio Code'"},
        "s": {"shell_command": "open -a 'Ghostty'"},
        "d": {"shell_command": "open -a 'Google Chrome'"},
    }

    manipulators = []
    for from_key, to_action in mappings.items():
        manipulators.extend(
            _generate_karabiner_simlayer_manipulators(
                layer_name=layer_name,
                layer_key=layer_key,
                from_key=from_key,
                to_action=to_action,
            )
        )

    karabiner_config = {
        "description": "Space simlayer: hjkl to arrow keys, asd to app shortcuts",
        "manipulators": manipulators,
    }

    out_path = "karabiner_simlayer_space_nav.json"
    with open(out_path, "w") as f:
        json.dump(karabiner_config, f, indent=4)

    print(
        f"Wrote Karabiner-Elements config to {out_path}. "
        "Add this under 'Complex Modifications' in the Karabiner-Elements app."
    )


if __name__ == "__main__":
    main()
