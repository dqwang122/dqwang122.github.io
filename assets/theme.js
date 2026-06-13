(function () {
  function systemTheme() {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') || systemTheme();
  }

  function updateToggleButton(theme) {
    var button = document.getElementById('theme-toggle');
    if (!button) {
      return;
    }

    var isDark = theme === 'dark';
    button.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
    button.setAttribute('title', isDark ? 'Switch to light mode' : 'Switch to dark mode');

    var icon = button.querySelector('i');
    if (icon) {
      icon.className = isDark ? 'fas fa-sun' : 'fas fa-moon';
    }
  }

  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    updateToggleButton(theme);
  }

  function toggleTheme() {
    setTheme(currentTheme() === 'dark' ? 'light' : 'dark');
  }

  document.addEventListener('DOMContentLoaded', function () {
    var button = document.getElementById('theme-toggle');
    if (!button) {
      return;
    }

    updateToggleButton(currentTheme());
    button.addEventListener('click', toggleTheme);
  });

  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () {
    if (!localStorage.getItem('theme')) {
      updateToggleButton(systemTheme());
    }
  });
})();
