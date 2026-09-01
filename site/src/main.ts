// Page entry. The preview upgrades the static image; if anything in the
// mount fails, the image stays and the page is still complete.

import "./styles/main.css";

import { mountPreview } from "./preview";

const host = document.getElementById("preview");
if (host !== null) {
  try {
    mountPreview(host);
  } catch (error) {
    console.error("preview unavailable, keeping the static image", error);
  }
}
