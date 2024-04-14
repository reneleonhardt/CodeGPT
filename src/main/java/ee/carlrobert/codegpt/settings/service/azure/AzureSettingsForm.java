package ee.carlrobert.codegpt.settings.service.azure;

import static ee.carlrobert.codegpt.credentials.Credential.getCredential;
import static ee.carlrobert.codegpt.credentials.Credential.sanitize;
import static ee.carlrobert.codegpt.credentials.CredentialsStore.CredentialKey.AZURE_ACTIVE_DIRECTORY_TOKEN;
import static ee.carlrobert.codegpt.credentials.CredentialsStore.CredentialKey.AZURE_OPENAI_API_KEY;
import static ee.carlrobert.codegpt.ui.UIUtil.withEmptyLeftBorder;

import com.intellij.ui.TitledSeparator;
import com.intellij.ui.components.JBPasswordField;
import com.intellij.ui.components.JBRadioButton;
import com.intellij.ui.components.JBTextField;
import com.intellij.util.ui.FormBuilder;
import com.intellij.util.ui.UI;
import ee.carlrobert.codegpt.CodeGPTBundle;
import java.util.List;
import java.util.Map;
import javax.swing.ButtonGroup;
import javax.swing.JPanel;
import org.jetbrains.annotations.Nullable;

public class AzureSettingsForm {

  private final JBRadioButton useAzureApiKeyAuthenticationRadioButton;
  private final JBPasswordField azureApiKeyField;
  private final JPanel azureApiKeyFieldPanel;
  private final JBRadioButton useAzureActiveDirectoryAuthenticationRadioButton;
  private final JBPasswordField azureActiveDirectoryTokenField;
  private final JPanel azureActiveDirectoryTokenFieldPanel;
  private final JBTextField azureResourceNameField;
  private final JBTextField azureDeploymentIdField;
  private final JBTextField azureApiVersionField;

  public AzureSettingsForm(AzureSettingsState settings) {
    useAzureApiKeyAuthenticationRadioButton = new JBRadioButton(
        CodeGPTBundle.get("settingsConfigurable.service.azure.useApiKeyAuth.label"),
        settings.isUseAzureApiKeyAuthentication());
    useAzureActiveDirectoryAuthenticationRadioButton = new JBRadioButton(
        CodeGPTBundle.get("settingsConfigurable.service.azure.useActiveDirectoryAuth.label"),
        settings.isUseAzureActiveDirectoryAuthentication());
    azureApiKeyField = new JBPasswordField();
    azureApiKeyField.setColumns(30);
    azureApiKeyField.setText(getCredential(AZURE_OPENAI_API_KEY));
    azureApiKeyFieldPanel = UI.PanelFactory.panel(azureApiKeyField)
        .withLabel(CodeGPTBundle.get("settingsConfigurable.shared.apiKey.label"))
        .resizeX(false)
        .createPanel();
    azureActiveDirectoryTokenField = new JBPasswordField();
    azureActiveDirectoryTokenField.setColumns(30);
    azureActiveDirectoryTokenField.setText(getCredential(AZURE_ACTIVE_DIRECTORY_TOKEN));
    azureActiveDirectoryTokenFieldPanel = UI.PanelFactory.panel(azureActiveDirectoryTokenField)
        .withLabel(CodeGPTBundle.get("settingsConfigurable.service.azure.bearerToken.label"))
        .resizeX(false)
        .createPanel();
    azureResourceNameField = new JBTextField(settings.getResourceName(), 35);
    azureDeploymentIdField = new JBTextField(settings.getDeploymentId(), 35);
    azureApiVersionField = new JBTextField(settings.getApiVersion(), 35);

    registerPanelsVisibility(settings);
    registerRadioButtons();
  }

  public JPanel getForm() {
    var authPanel = withEmptyLeftBorder(FormBuilder.createFormBuilder()
        .addComponent(UI.PanelFactory
            .panel(useAzureApiKeyAuthenticationRadioButton)
            .resizeX(false)
            .createPanel())
        .addComponent(withEmptyLeftBorder(azureApiKeyFieldPanel))
        .addComponent(UI.PanelFactory
            .panel(useAzureActiveDirectoryAuthenticationRadioButton)
            .resizeX(false)
            .createPanel())
        .addComponent(withEmptyLeftBorder(azureActiveDirectoryTokenFieldPanel))
        .getPanel());

    var configPanel = withEmptyLeftBorder(UI.PanelFactory.grid()
        .add(UI.PanelFactory.panel(azureResourceNameField)
            .withLabel(CodeGPTBundle.get(
                "settingsConfigurable.service.azure.resourceName.label"))
            .resizeX(false)
            .withComment(CodeGPTBundle.get(
                "settingsConfigurable.service.azure.resourceName.comment")))
        .add(UI.PanelFactory.panel(azureDeploymentIdField)
            .withLabel(CodeGPTBundle.get(
                "settingsConfigurable.service.azure.deploymentId.label"))
            .resizeX(false)
            .withComment(CodeGPTBundle.get(
                "settingsConfigurable.service.azure.deploymentId.comment")))
        .add(UI.PanelFactory.panel(azureApiVersionField)
            .withLabel(CodeGPTBundle.get(
                "shared.apiVersion"))
            .resizeX(false)
            .withComment(CodeGPTBundle.get(
                "settingsConfigurable.service.azure.apiVersion.comment")))
        .createPanel());

    return FormBuilder.createFormBuilder()
        .addComponent(new TitledSeparator(
            CodeGPTBundle.get("settingsConfigurable.shared.authentication.title")))
        .addComponent(authPanel)
        .addComponent(new TitledSeparator(
            CodeGPTBundle.get("settingsConfigurable.shared.requestConfiguration.title")))
        .addComponent(configPanel)
        .addComponentFillVertically(new JPanel(), 0)
        .getPanel();
  }

  public AzureSettingsState getCurrentState() {
    var state = new AzureSettingsState();
    state.setUseAzureActiveDirectoryAuthentication(
        useAzureActiveDirectoryAuthenticationRadioButton.isSelected());
    state.setUseAzureApiKeyAuthentication(useAzureApiKeyAuthenticationRadioButton.isSelected());
    state.setResourceName(azureResourceNameField.getText());
    state.setDeploymentId(azureDeploymentIdField.getText());
    state.setApiVersion(azureApiVersionField.getText());
    return state;
  }

  public void resetForm() {
    var state = AzureSettings.getCurrentState();
    azureApiKeyField.setText(getCredential(AZURE_OPENAI_API_KEY));
    azureActiveDirectoryTokenField.setText(getCredential(AZURE_ACTIVE_DIRECTORY_TOKEN));
    useAzureApiKeyAuthenticationRadioButton.setSelected(state.isUseAzureApiKeyAuthentication());
    useAzureActiveDirectoryAuthenticationRadioButton.setSelected(
        state.isUseAzureActiveDirectoryAuthentication());
    azureResourceNameField.setText(state.getResourceName());
    azureDeploymentIdField.setText(state.getDeploymentId());
    azureApiVersionField.setText(state.getApiVersion());
  }

  public @Nullable String getActiveDirectoryToken() {
    return sanitize(new String(azureActiveDirectoryTokenField.getPassword()));
  }

  public JBPasswordField getActiveDirectoryTokenField() {
    return azureActiveDirectoryTokenField;
  }

  public @Nullable String getApiKey() {
    return sanitize(new String(azureApiKeyField.getPassword()));
  }

  public JBPasswordField getApiKeyField() {
    return azureApiKeyField;
  }

  private void registerPanelsVisibility(AzureSettingsState azureSettings) {
    azureApiKeyFieldPanel.setVisible(azureSettings.isUseAzureApiKeyAuthentication());
    azureActiveDirectoryTokenFieldPanel.setVisible(
        azureSettings.isUseAzureActiveDirectoryAuthentication());
  }

  private void registerRadioButtons() {
    registerRadioButtons(List.of(
        Map.entry(useAzureApiKeyAuthenticationRadioButton, azureApiKeyFieldPanel),
        Map.entry(useAzureActiveDirectoryAuthenticationRadioButton,
            azureActiveDirectoryTokenFieldPanel)));
  }

  // TODO: Move
  private void registerRadioButtons(List<Map.Entry<JBRadioButton, JPanel>> entries) {
    var buttonGroup = new ButtonGroup();
    entries.forEach(entry -> buttonGroup.add(entry.getKey()));
    entries.forEach(entry -> entry.getKey().addActionListener((e) -> {
      for (Map.Entry<JBRadioButton, JPanel> innerEntry : entries) {
        innerEntry.getValue().setVisible(innerEntry.equals(entry));
      }
    }));
  }
}
