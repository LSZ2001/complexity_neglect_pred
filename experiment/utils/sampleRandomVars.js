    // Functions for sampling random variables
    function sampleRandomInt(min, max) {
        min = Math.ceil(min);
        max = Math.floor(max);
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }
    function sampleUniform(min, max) {
        return Math.random() * (max - min) + min;
    }
    function sampleGamma(shape) {
        if (shape < 1) {
            // Use boost method for shape < 1
            const u = Math.random();
            return sampleGamma(1 + shape) * Math.pow(u, 1 / shape);
        }
        const d = shape - 1 / 3;
        const c = 1 / Math.sqrt(9 * d);
        while (true) {
            let x, v;
            do {
            x = sampleGaussian(0,1); // standard normal
            v = 1 + c * x;
            } while (v <= 0);
            v = v * v * v;
            const u = Math.random();
            if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
            if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
        }
    }
    // Box-Muller transform for standard normal
    function sampleGaussian(mean, stdDev) {
        let u1 = 0, u2 = 0;
        // Convert [0,1) to (0,1)
        while (u1 === 0) u1 = Math.random();
        while (u2 === 0) u2 = Math.random();
        const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
    }
    function sampleBeta(alpha, beta) {
        const x = sampleGamma(alpha);
        const y = sampleGamma(beta);
        return x / (x + y);
    }

    function extractOnesIndices(A) {
        return A.map(row =>
            row.reduce((indices, val, idx) => {
            if (val === 1) indices.push(idx);
            return indices;
            }, [])
        );
    }

    function isAffordable(state_alltrials, Affordances){
        /* state_alltrials: a K*J binary matrix of states (K problems/trials, each defined by the presences of J objects)
           Affordances: a I*J binary matrix of affordances (I solutions, J objects)
           output: a K*I binary matrix, denoting whether Solution i is affordable in Trial k.
        */
        var num_trials = state_alltrials.length;
        var num_relevantobjs = Affordances[0].length;
        var solutions_affordability_alltrials = [];
        for (let k=0; k<num_trials; k++){ // For each problem
            var solutions_affordability = [];
            for (let i=0; i<Affordances.length; i++){ // Is Solution a_i affordable under the current problem, state_true?
                var temp = 0
                for (let j=0; j<num_relevantobjs; j++){
                    temp += ((1-state_alltrials[k][j]) * Affordances[i][j])
                }
                solutions_affordability.push(Number(temp==0));
            }
            solutions_affordability_alltrials.push(solutions_affordability);
        }
        return solutions_affordability_alltrials
    }


function downloadCSV(csv, filename) {
  // Retrive csv file from task
  var csvFile = new Blob( [csv], {type: "text/csv"});
  // Download link
  var downloadlink = document.createElement("a");
  // Download link download
  downloadlink.download = filename;
  downloadlink.href = window.URL.createObjectURL(csvFile);
  downloadlink.style.display = 'none';
  document.body.appendChild(downloadlink)
  downloadlink.click()
}



// Circular arrangment of stimulus images
function generateDisplayLocs(n_stimuli, circle_diameter, image_size) {
    // circle params
    var diam = circle_diameter; // pixels
    var radi = diam / 2;
    var paper_size = diam + image_size[0];
    // stimuli width, height
    var stimh = image_size[0];
    var stimw = image_size[1];
    var hstimh = stimh / 2;
    var hstimw = stimw / 2;
    var display_locs = [];
    var random_offset = Math.floor(Math.random() * 360);
    for (var i = 0; i < n_stimuli; i++) {
        display_locs.push([
            Math.floor(paper_size / 2 + this.cosd(random_offset + i * (360 / n_stimuli)) * radi - hstimw),
            Math.floor(paper_size / 2 - this.sind(random_offset + i * (360 / n_stimuli)) * radi - hstimh),
        ]);
    }
    return display_locs;
}

function cosd(num) {
    return Math.cos((num / 180) * Math.PI);
}
function sind(num) {
    return Math.sin((num / 180) * Math.PI);
}

function only_giveup_affordable(isaffordable_trialsolution_trial){
    for (let i = 0; i < (isaffordable_trialsolution_trial.length - 1); i++) {
        if (isaffordable_trialsolution_trial[i] > 0.5) {
            return false; // If any element before the last is 1, return false
        }
    }
    return true; // All elements before the last are 0
}


function makeHalfArray(N, val0, val1) {
/* Generate an array with length N, such that:
    1) Half of its entries are val0, and the other half is val1;
    2) The first entry is val1;
    3) The other entries are randomly assigned values, but 1) still holds.
*/
  if (N % 2 !== 0) {
    throw new Error("N must be even for an exact half/half split");
  }

  const half = N / 2;              // how many of each value we want
  const remainingHigh = half - 1;  // we’ll use one 0.8 at index 0
  const remainingLow  = half;

  // 1) build the “rest” of the array
  const rest = [
    ...Array(remainingHigh).fill(val1),
    ...Array(remainingLow) .fill(val0)
  ];

  // 2) shuffle it in-place (Fisher–Yates)
  for (let i = rest.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [rest[i], rest[j]] = [rest[j], rest[i]];
  }

  // 3) prepend the fixed first element
  return [val1, ...rest];
}



function createIntegerRange(a, b) {
  return Array.from({ length: b - a }, (_, index) => a + index);
}