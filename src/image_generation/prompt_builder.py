from typing import Dict


def generate_base_prompt(area_stats: Dict[int, dict], threshold: float = 2.0) -> str:
    """Create a simple, structured base prompt from area percentages.

    Rules:
    - ignore classes below `threshold` percent
    - ignore Background
    - list dominant classes first
    """
    significant = [
        (v["name"], v["percentage"]) for v in area_stats.values()
        if v["percentage"] >= threshold and v["name"] != "Background"
    ]
    if not significant:
        return "A satellite view of an area."

    phrases = []
    for name, pct in significant:
        n = name
        p = pct
        if n == "Residential":
            if p > 50:
                phrases.append(f"predominantly dense residential housing ({p:.0f}%)")
            elif p > 20:
                phrases.append(f"residential neighborhoods ({p:.0f}%)")
            else:
                phrases.append(f"scattered residential areas ({p:.0f}%)")

        elif n == "Road":
            if p > 20:
                phrases.append(f"dense road network and urban infrastructure ({p:.0f}%)")
            elif p > 5:
                phrases.append(f"road network and arterial streets ({p:.0f}%)")
            else:
                phrases.append(f"light road infrastructure ({p:.0f}%)")

        elif n == "River":
            if p > 15:
                phrases.append(f"major river and water bodies ({p:.0f}%)")
            else:
                phrases.append(f"small water channels ({p:.0f}%)")

        elif n == "Forest":
            if p > 30:
                phrases.append(f"dense forest cover ({p:.0f}%)")
            else:
                phrases.append(f"sparse vegetation and tree cover ({p:.0f}%)")

        elif n == "Unused Land":
            if p > 20:
                phrases.append(f"large undeveloped bare land ({p:.0f}%)")
            else:
                phrases.append(f"patches of unused open land ({p:.0f}%)")

        elif n == "Agriculture":
            if p > 30:
                phrases.append(f"extensive agricultural fields ({p:.0f}%)")
            else:
                phrases.append(f"small agricultural patches ({p:.0f}%)")

        else:
            phrases.append(f"{n.lower()} ({p:.0f}%)")

    return "A satellite image showing " + ", ".join(phrases) + "."


def build_final_prompt(base_prompt: str, custom_prompt: str = "", strategy: str = "append") -> str:
    if strategy == "override" and custom_prompt:
        return custom_prompt
    if strategy == "append" and custom_prompt:
        return base_prompt + " " + custom_prompt
    return base_prompt
