
/**
 * @jest-environment jsdom
 */

import { JSDOM } from "jsdom";

// Simulate DOM
document.body.innerHTML = `
  <select id="team1-select">
    <option value="lakers">Lakers</option>
    <option value="nets">Nets</option>
  </select>
  <div id="team1-circles">
    <div class="circle"></div>
    <div class="circle"></div>
  </div>
`;

const teamLogos = {
  lakers: "https://upload.wikimedia.org/wikipedia/commons/3/3c/Los_Angeles_Lakers_logo.svg",
  nets: "https://upload.wikimedia.org/wikipedia/commons/4/44/Brooklyn_Nets_newlogo.svg",
};

function updateTeamCircles(teamSelectId, circlesContainerId) {
  const select = document.getElementById(teamSelectId);
  const container = document.getElementById(circlesContainerId);
  const circles = container.querySelectorAll('.circle');

  select.addEventListener('change', () => {
    const team = select.value;
    const logo = teamLogos[team];

    circles.forEach(circle => {
      if (logo) {
        circle.style.backgroundColor = "transparent";
        circle.style.backgroundImage = `url(${logo})`;
        circle.style.border = "none";
      } else {
        circle.style.backgroundImage = "none";
        circle.style.backgroundColor = "#1D428A";
        circle.style.border = "2px solid white";
      }
    });
  });
}

updateTeamCircles("team1-select", "team1-circles");

test("updates circle backgrounds based on team select", () => {
  const select = document.getElementById("team1-select");
  select.value = "nets";

  const event = new Event("change");
  select.dispatchEvent(event);

  const circles = document.querySelectorAll(".circle");
  circles.forEach(circle => {
    expect(circle.style.backgroundImage).toContain(teamLogos["nets"]);
    expect(circle.style.backgroundColor).toBe("transparent");
    expect(circle.style.border).toBe("none");
  });
});