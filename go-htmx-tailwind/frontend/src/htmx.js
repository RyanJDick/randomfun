import htmx from "htmx.org"

// This file is a .js file insted of a .ts file because HTMX is written in
// javascript. This is simpler than having to write a bunch of typescript
// boilerplate to be able to use HTMX in a .ts file.
window.htmx = htmx;
