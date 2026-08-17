---
layout: page-head
permalink: /projects/
title: Projects
body_class: projects-page
---

<h1 id="projects-top">Research Projects</h1>

<p class="projects-intro">Selected research systems, benchmarks, datasets, and open-source implementations.</p>

<nav class="project-group-nav" aria-label="Project areas">
  {%- for group in site.data.project_groups %}
  <a href="#{{ group.id | escape }}">
    <i class="{{ group.icon | escape }}" aria-hidden="true"></i>
    {{ group.label | escape }}
  </a>
  {%- endfor %}
</nav>

{%- for group in site.data.project_groups %}
  {%- assign group_projects = site.data.pubs | where: "project_group", group.id | sort: "project_order" %}
  {%- if group_projects.size > 0 %}
<section class="project-group" id="{{ group.id | escape }}">
  <header class="project-group-header">
    <div>
      <h2><i class="{{ group.icon | escape }}" aria-hidden="true"></i> {{ group.label | escape }}</h2>
      <p>{{ group.description | escape }}</p>
    </div>
    <a class="project-back" href="#projects-top" aria-label="Back to top">Top &#8648;</a>
  </header>

  <div class="project-list">
    {%- for project in group_projects %}
    <article class="project-item{% unless project.image %} project-item-no-image{% endunless %}">
      {%- if project.image %}
      <button class="project-figure"
              type="button"
              onclick="zoomImage(this.querySelector('img'))"
              aria-label="Zoom the {{ project.project_name | escape }} figure">
        <img src="{{ project.image | escape }}" alt="{{ project.image_alt | escape }}">
      </button>
      {%- endif %}

      <div class="project-details">
        <div class="project-meta">
          <span class="project-venue">{{ project.venue | escape }}</span>
          {%- for topic in project.topics %}
            {%- assign topic_label = site.data.publication_topics[topic] | default: topic %}
          <span class="project-topic">{{ topic_label | escape }}</span>
          {%- endfor %}
        </div>

        <h3>{{ project.project_name | escape }}</h3>
        <p class="project-title">{{ project.title | escape }}</p>
        <p class="project-summary">{{ project.project_summary | escape }}</p>
        <p class="project-authors">{{ project.authors_html }}</p>

        <nav class="project-links" aria-label="Links for {{ project.project_name | escape }}">
          {%- for link in project.links %}
          <a href="{{ link.url | escape }}"><i class="{{ link.icon | escape }}" aria-hidden="true"></i> {{ link.label | escape }}</a>
          {%- endfor %}
          {%- for link in project.project_links %}
          <a href="{{ link.url | escape }}"><i class="{{ link.icon | escape }}" aria-hidden="true"></i> {{ link.label | escape }}</a>
          {%- endfor %}
        </nav>
      </div>
    </article>
    {%- endfor %}
  </div>
</section>
  {%- endif %}
{%- endfor %}
