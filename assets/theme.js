(function () {
  function systemTheme() {
    if (!window.matchMedia) {
      return 'light';
    }

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
  }

  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    updateToggleButton(theme);
  }

  function toggleTheme() {
    setTheme(currentTheme() === 'dark' ? 'light' : 'dark');
  }

  function initPublicationFilters() {
    var filterBar = document.querySelector('.publication-filters');
    if (!filterBar) {
      return;
    }

    var buttons = filterBar.querySelectorAll('[data-publication-filter]');
    var items = document.querySelectorAll('.publication-item[data-topics]');
    var groups = document.querySelectorAll('[data-publication-group]');
    var status = document.getElementById('publication-filter-status');

    function applyFilter(filter, filterTopics, label) {
      var visibleCount = 0;

      Array.prototype.forEach.call(items, function (item) {
        var topics = (item.getAttribute('data-topics') || '').split(/\s+/);
        var isVisible = filter === 'all' || filterTopics.some(function (topic) {
          return topics.indexOf(topic) !== -1;
        });
        item.hidden = !isVisible;

        if (isVisible) {
          visibleCount += 1;
        }
      });

      Array.prototype.forEach.call(groups, function (group) {
        group.hidden = !group.querySelector('.publication-item:not([hidden])');
      });

      Array.prototype.forEach.call(buttons, function (button) {
        button.setAttribute(
          'aria-pressed',
          button.getAttribute('data-publication-filter') === filter ? 'true' : 'false'
        );
      });

      if (status) {
        status.textContent = filter === 'all'
          ? 'Showing all ' + visibleCount + ' publications.'
          : 'Showing ' + visibleCount + ' ' + label + ' publications.';
      }
    }

    Array.prototype.forEach.call(buttons, function (button) {
      button.addEventListener('click', function () {
        var filterTopics = (button.getAttribute('data-publication-topics') || '')
          .split(/\s+/)
          .filter(Boolean);

        applyFilter(
          button.getAttribute('data-publication-filter'),
          filterTopics,
          button.textContent.trim()
        );
      });
    });

    applyFilter('all', [], 'All');
  }

  function initPageControls() {
    initPublicationFilters();

    var button = document.getElementById('theme-toggle');
    if (button) {
      updateToggleButton(currentTheme());
      button.addEventListener('click', toggleTheme);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPageControls);
  } else {
    initPageControls();
  }

  var colorSchemeQuery = window.matchMedia
    ? window.matchMedia('(prefers-color-scheme: dark)')
    : null;
  var handleSystemThemeChange = function () {
    if (!localStorage.getItem('theme')) {
      updateToggleButton(systemTheme());
    }
  };

  if (colorSchemeQuery && colorSchemeQuery.addEventListener) {
    colorSchemeQuery.addEventListener('change', handleSystemThemeChange);
  } else if (colorSchemeQuery && colorSchemeQuery.addListener) {
    colorSchemeQuery.addListener(handleSystemThemeChange);
  }

  window.zoomImage = function (img) {
    if (!img) {
      return;
    }

    var existingModal = document.querySelector('.image-zoom-modal');
    if (existingModal) {
      existingModal.remove();
    }

    var previousOverflow = document.body.style.overflow;
    var modal = document.createElement('div');
    var zoomedImg = document.createElement('img');
    var closeButton = document.createElement('button');

    modal.className = 'image-zoom-modal';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.setAttribute('aria-label', img.alt ? 'Expanded image: ' + img.alt : 'Expanded image');
    modal.tabIndex = -1;

    zoomedImg.src = img.currentSrc || img.src;
    zoomedImg.alt = img.alt || '';

    closeButton.className = 'image-zoom-close';
    closeButton.type = 'button';
    closeButton.setAttribute('aria-label', 'Close expanded image');
    closeButton.innerHTML = '&times;';

    function closeModal() {
      document.removeEventListener('keydown', handleKeydown);
      document.body.style.overflow = previousOverflow;
      modal.remove();
    }

    function handleKeydown(event) {
      if (event.key === 'Escape') {
        closeModal();
      }
    }

    modal.addEventListener('click', function (event) {
      if (event.target === modal || event.target === zoomedImg) {
        closeModal();
      }
    });
    closeButton.addEventListener('click', closeModal);
    document.addEventListener('keydown', handleKeydown);

    modal.appendChild(zoomedImg);
    modal.appendChild(closeButton);
    document.body.appendChild(modal);
    document.body.style.overflow = 'hidden';
    modal.focus();
  };
})();
