// Navbar active state on scroll
window.addEventListener("scroll", () => {
  let sections = document.querySelectorAll("section");
  let navLinks = document.querySelectorAll(".nav-link");

  let current = "";
  sections.forEach(section => {
    let sectionTop = section.offsetTop - 80;
    if (scrollY >= sectionTop) {
      current = section.getAttribute("id");
    }
  });

  navLinks.forEach(link => {
    link.classList.remove("active");
    if (link.getAttribute("href").includes(current)) {
      link.classList.add("active");
    }
  });
});

// Fade-in animations
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add("fade-in");
    }
  });
}, { threshold: 0.2 });

document.querySelectorAll("section").forEach(sec => observer.observe(sec));

// Gallery filter (example)
function filterGallery(type) {
  let items = document.querySelectorAll(".pattern-card");
  items.forEach(item => {
    if (type === "all" || item.dataset.type === type) {
      item.style.display = "block";
    } else {
      item.style.display = "none";
    }
  });
}
