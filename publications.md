---
layout: page-head
permalink: /publications/
title: Publications
body_class: publications-page
---

# Publications

<p class="publications-intro">A complete citation record is available on <a href="https://scholar.google.com/citations?hl=zh-CN&amp;user=mAo_lUwAAAAJ">Google Scholar</a>.</p>

<!-- <nav class="publication-views" aria-label="Alternative publication views"> -->
  <!-- <a href="/publications/years/"><i class="fas fa-calendar-alt" aria-hidden="true"></i> Year index</a> -->
  <!-- <a href="/publications/tags/"><i class="fas fa-tags" aria-hidden="true"></i> Topic index</a> -->
  <!-- <a href="/publications/confs/"><i class="fas fa-graduation-cap" aria-hidden="true"></i> Conference index</a> -->
<!-- </nav> -->

<div class="publication-filters" role="group" aria-label="Filter publications by topic">
  {%- for filter in site.data.publication_filters %}
  <button type="button"
          data-publication-filter="{{ filter.id | escape }}"
          data-publication-topics="{{ filter.topics | join: ' ' | escape }}"
          aria-pressed="{% if filter.id == 'all' %}true{% else %}false{% endif %}">{{ filter.label | escape }}</button>
  {%- endfor %}
</div>
<p id="publication-filter-status" class="publication-filter-status" aria-live="polite"></p>

{%- for section in site.data.publication_sections %}
  {%- assign section_publications = site.data.pubs | where: "section", section.id %}
  {%- if section_publications.size > 0 %}
<section class="publication-group" data-publication-group>
  <h2>{{ section.label | escape }}</h2>
  <div class="publication-list">
    {%- for pub in section_publications %}
    <article class="publication-item{% unless pub.image %} publication-item-no-image{% endunless %}"
             data-topics="{{ pub.topics | join: ' ' | escape }}">
      <div class="publication-details">
        <div class="publication-meta">
          <span class="publication-venue">{{ pub.venue | escape }}</span>
          {%- for topic in pub.topics %}
            {%- assign topic_label = site.data.publication_topics[topic] | default: topic %}
          <span class="publication-topic">{{ topic_label | escape }}</span>
          {%- endfor %}
        </div>
        <h3>{{ pub.title | escape }}</h3>
        <p class="publication-authors">{{ pub.authors_html }}</p>
        {%- if pub.links.size > 0 %}
        <nav class="publication-links" aria-label="Links for {{ pub.title | escape }}">
          {%- for link in pub.links %}
          <a href="{{ link.url | escape }}"><i class="{{ link.icon | escape }}" aria-hidden="true"></i> {{ link.label | escape }}</a>
          {%- endfor %}
        </nav>
        {%- endif %}
      </div>
      {%- if pub.image %}
      <button class="publication-figure"
              type="button"
              onclick="zoomImage(this.querySelector('img'))"
              aria-label="Zoom the {{ pub.title | escape }} figure">
        <img src="{{ pub.image | escape }}" alt="{{ pub.image_alt | escape }}">
      </button>
      {%- endif %}
    </article>
    {%- endfor %}
  </div>
</section>
  {%- endif %}
{%- endfor %}

<p class="publication-note">* Equal contribution.</p>
