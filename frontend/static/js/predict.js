// predict.js

const teamLogos = {
    lakers: "https://upload.wikimedia.org/wikipedia/commons/3/3c/Los_Angeles_Lakers_logo.svg",
    nets: "https://upload.wikimedia.org/wikipedia/commons/4/44/Brooklyn_Nets_newlogo.svg",
    bucks: "https://upload.wikimedia.org/wikipedia/en/1/1a/Milwaukee_Bucks_logo.svg",
    celtics: "https://upload.wikimedia.org/wikipedia/en/8/8f/Boston_Celtics.svg",
    heat: "https://upload.wikimedia.org/wikipedia/en/f/fb/Miami_Heat_logo.svg",
    warriors: "https://upload.wikimedia.org/wikipedia/en/0/01/Golden_State_Warriors_logo.svg"
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
          circle.style.backgroundColor = "transparent";  // remove blue background
          circle.style.backgroundImage = `url(${logo})`;
          circle.style.border = "none";  // remove border for logo look
        } else {
          circle.style.backgroundImage = "none";
          circle.style.backgroundColor = "#1D428A";
          circle.style.border = "2px solid white";
        }
      });
    });
  }
  
  updateTeamCircles('team1-select', 'team1-circles');
  updateTeamCircles('team2-select', 'team2-circles');
  