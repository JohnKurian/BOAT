import logo from './logo.svg';
import React, { useState, useEffect, Component } from 'react';
import './App.css';
import Home from './Home.js'
import CreateExperiment from './CreateExperiment.js'
import SelectExperiment from './SelectExperiment.js'
import Experiment from './Experiment.js'
import Run from './Run.js'
import history from './history';

import {
  BrowserRouter as Router,
  Switch,
  Route,
  Link,
  Redirect
} from "react-router-dom";


class App extends Component {

  constructor() {
    super();
    this.state = {
      user: null
    };
  }


  render() {
  return (
    <div className="App">

        <Router>

        <div>
        {/* A <Switch> looks through its children <Route>s and
            renders the first one that matches the current URL. */}
        <Switch>
          <Route exact path="/" render={() => (<Home/>)} />
          <Route exact path="/create_exp" render={() => (<CreateExperiment/>)} />
          <Route exact path="/experiments" render={() => (<SelectExperiment  {...this.props} history = {history}/>)} />
          <Route exact path="/experiments/:experiment_id" render={() => (<Experiment  {...this.props} history = {history}/>)} />
          <Route exact path="/runs/:run_id" render={() => (<Run  {...this.props} history = {history}/>)} />

        </Switch>
      </div>
    </Router>
    </div>
  );
          }
}




export default App;