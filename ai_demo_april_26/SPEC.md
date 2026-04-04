# Restaurant Card — Build Spec

Build a single-page website that displays a **Google Maps / Yelp-style restaurant card** for a fictional taco shop called **"Casa del Sol Taqueria"**.

## Requirements

### Layout
- A single card, centered on the page with a subtle background
- Card width: ~400px, with rounded corners and a soft drop shadow

### Card Contents (top to bottom)
1. **Hero image** — use a placeholder from `https://images.unsplash.com/photo-1565299585323-38d6b0865b47?w=400&h=250&fit=crop` (tacos photo)
2. **Restaurant name** — "Casa del Sol Taqueria" in bold
3. **Rating row** — 4.5 stars (filled/half-filled star icons using unicode ★/☆ or SVG), with "(1,284 reviews)" next to it
4. **Tags row** — pill-shaped tags: "Mexican", "Tacos", "$$"
5. **Info rows** with small icons or emoji:
   - 🕐 "Open now · Closes 10 PM"
   - 📍 "0.3 mi · Mission District, SF"
   - 📞 "(415) 555-0142"
6. **Action buttons row** — two buttons side by side:
   - "Directions" (primary/filled style)
   - "Save" (outline/secondary style)

### Styling
- Clean, modern look — think Google Maps card aesthetic
- Use a nice sans-serif font (e.g. Google Fonts)
- Warm color palette (oranges/terracotta accents)
- Subtle hover effects on the buttons
- Mobile-friendly

### Tech
- Single `index.html` file (inline CSS and no JS needed)
- No frameworks, no build step
- Should look polished when opened directly in a browser
